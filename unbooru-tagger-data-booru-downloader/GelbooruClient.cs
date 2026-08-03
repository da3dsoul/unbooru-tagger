using System.Globalization;
using System.Net;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace UnbooruTagger.Crawler;

/// <summary>
/// Thin Gelbooru (<c>dapi</c>) API client. Always requests <c>json=1</c> explicitly —
/// the default response shape varies by endpoint (some are XML-only) and posts arrive
/// wrapped in a <c>"post"</c> key rather than a bare array the way Danbooru's are.
/// Tag names and each post's tag string are HTML-entity-encoded and must be decoded
/// (<c>WebUtility.HtmlDecode</c>) or stray entities like <c>&amp;amp;</c> would silently
/// fragment what should be one canonical tag into two — a real gotcha found in
/// <c>BooruSharp</c>'s Gelbooru template, which does the same decoding. <c>created_at</c>
/// is not ISO-8601; it's the C-style <c>asctime</c> shape, parsed with an exact custom
/// format string (also taken from <c>BooruSharp</c>) rather than a naive <c>DateTime.Parse</c>,
/// which would throw on it.
/// </summary>
public sealed class GelbooruClient(HttpClient http, IRateLimiter rateLimiter, string? apiKey = null, string? userId = null, string baseUrl = "https://gelbooru.com")
    : BooruHttpClientBase(http, rateLimiter), IBooruClient
{
    private const string GelbooruDateFormat = "ddd MMM dd HH:mm:ss zzz yyyy";
    private readonly string _baseUrl = baseUrl.TrimEnd('/');

    public string SiteName => "gelbooru";

    /// <summary>Gelbooru hard-caps posts at 100/request regardless of the requested <c>limit</c>.</summary>
    public int PageSize => 100;

    private string AuthQuery() =>
        apiKey is not null && userId is not null
            ? $"&api_key={Uri.EscapeDataString(apiKey)}&user_id={Uri.EscapeDataString(userId)}"
            : "";

    public async IAsyncEnumerable<BooruTagCount> ListTagsByCountDescendingAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        for (var pid = 0; ; pid++)
        {
            // orderby=count picks the sort field; order=DESC is the separate, required
            // direction param — Gelbooru's dapi docs split the two, and orderby alone
            // silently falls back to an unspecified order, breaking the "sorted
            // descending, stop once under --min-images" assumption TagSurveyor relies on.
            var uri = new Uri($"{_baseUrl}/index.php?page=dapi&s=tag&q=index&json=1&orderby=count&order=DESC&limit=100&pid={pid}{AuthQuery()}");
            var json = await GetStringAsync(uri, cancellationToken).ConfigureAwait(false);
            using var doc = JsonDocument.Parse(json);

            if (!doc.RootElement.TryGetProperty("tag", out var tagsElement) || tagsElement.ValueKind != JsonValueKind.Array || tagsElement.GetArrayLength() == 0)
                yield break;

            foreach (var element in tagsElement.EnumerateArray())
            {
                var category = element.TryGetProperty("type", out var typeProp) && typeProp.TryGetInt32(out var typeCode)
                    ? TagCategoryNaming.FromRawCode(typeCode)
                    : TagCategory.General;
                yield return new BooruTagCount(
                    TagCategoryNaming.Identity(WebUtility.HtmlDecode(element.GetProperty("name").GetString()!), category),
                    element.GetProperty("count").GetInt32());
            }
        }
    }

    /// <summary>
    /// Gelbooru's own row shape on its (HTML-only — no <c>dapi</c> equivalent exists;
    /// <c>s=tag_alias</c>/<c>s=alias</c> under <c>page=dapi</c> both come back empty)
    /// alias listing page: <c>&lt;td&gt;&lt;a href="...tags=ANTECEDENT"&gt;display&lt;/a&gt;
    /// &lt;span class="tag-count"&gt;N&lt;/span&gt; &lt;b&gt;&amp;rarr;&lt;/b&gt;
    /// &lt;a href="...tags=CONSEQUENT"&gt;display&lt;/a&gt; &lt;span class="tag-count"&gt;N&lt;/span&gt;&lt;/td&gt;</c>
    /// per row. The raw tag name is read from each link's own <c>tags=</c> query
    /// parameter, not its display text (which has underscores rendered back out as
    /// spaces) — the same underscored form <see cref="ListPostsAsync"/> expects.
    /// </summary>
    private static readonly Regex AliasRowPattern = new(
        """<td><a href="[^"]*?tags=([^"&]+)"[^>]*>[^<]*</a>\s*<span class="tag-count">\d+</span>\s*<b>&rarr;</b>\s*<a href="[^"]*?tags=([^"&]+)"[^>]*>[^<]*</a>\s*<span class="tag-count">\d+</span></td>""",
        RegexOptions.Compiled);

    /// <summary>How much <c>pid</c> advances between alias-listing pages — the step the site's own "next page" links use (confirmed against a live fetch), not a documented page size.</summary>
    private const int AliasPageStep = 50;

    private static string DecodeTagName(string value) => WebUtility.HtmlDecode(Uri.UnescapeDataString(value));

    /// <summary>
    /// Streams Gelbooru's own active tag-alias table by scraping <c>page=alias&amp;s=list</c>
    /// (there is no JSON <c>dapi</c> equivalent — see <see cref="AliasRowPattern"/>'s own
    /// doc comment). Danbooru's <see cref="DanbooruClient.ListActiveTagAliasesAsync"/>
    /// doc comment used to claim Gelbooru had no alias listing at all and that Danbooru's
    /// table alone was enough since a raw name from either site could be looked up
    /// against it — that's wrong: Gelbooru maintains its own, independently-diverging
    /// alias table (e.g. Danbooru's <c>nude_male_clothed_female</c> is Gelbooru's own
    /// <c>clothed_female_nude_male</c>; a "deprecated" Gelbooru tag like
    /// <c>curvy_figure</c> aliases to <c>curvy</c> with no Danbooru-side equivalent at
    /// all), invisible to a search from the OTHER site's canonical spelling. Searching a
    /// raw name Gelbooru itself has aliased away silently returns posts under Gelbooru's
    /// own current tag instead — same failure mode as an unknown Danbooru alias, just
    /// from the other direction, and only fixable by knowing Gelbooru's own table too.
    /// </summary>
    public async IAsyncEnumerable<BooruTagAlias> ListActiveTagAliasesAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        for (var pid = 0; ; pid += AliasPageStep)
        {
            var uri = new Uri($"{_baseUrl}/index.php?page=alias&s=list&pid={pid}");
            var html = await GetStringAsync(uri, cancellationToken).ConfigureAwait(false);

            var matches = AliasRowPattern.Matches(html);
            if (matches.Count == 0)
                yield break;

            foreach (Match match in matches)
                yield return new BooruTagAlias(DecodeTagName(match.Groups[1].Value), DecodeTagName(match.Groups[2].Value));
        }
    }

    public async Task<BooruPostPage> ListPostsAsync(string tagQuery, string? cursor, CancellationToken cancellationToken = default)
    {
        // Anchored on the last post id seen (like Danbooru's page=bN), not a raw page
        // offset (pid) — Gelbooru is a live, constantly-growing site, so a page-number
        // cursor is unsafe across a resumed crawl: every post added between the last run
        // and this one shifts what "page 5" means, silently skipping posts that moved
        // past page 5 or reprocessing ones that moved back to it. id:< combined with an
        // explicit sort:id:desc keeps every page anchored to real post ids, immune to
        // however many posts get added in between — same trick, same reasoning as
        // DanbooruClient's own cursor.
        var idFilter = cursor is null ? "" : $" id:<{cursor}";
        var uri = new Uri($"{_baseUrl}/index.php?page=dapi&s=post&q=index&json=1&tags={Uri.EscapeDataString(tagQuery + " sort:id:desc" + idFilter)}&limit={PageSize}{AuthQuery()}");
        var json = await GetStringAsync(uri, cancellationToken).ConfigureAwait(false);
        using var doc = JsonDocument.Parse(json);

        var posts = new List<BooruPost>();
        var rawCount = 0;
        long? lastRawId = null;
        if (doc.RootElement.TryGetProperty("post", out var postsElement) && postsElement.ValueKind == JsonValueKind.Array)
        {
            foreach (var element in postsElement.EnumerateArray())
            {
                rawCount++;
                lastRawId = element.GetProperty("id").GetInt64();

                if (!element.TryGetProperty("md5", out var md5Prop) || md5Prop.ValueKind != JsonValueKind.String)
                    continue;
                if (!element.TryGetProperty("file_url", out var fileUrlProp) || fileUrlProp.ValueKind != JsonValueKind.String
                    || string.IsNullOrEmpty(fileUrlProp.GetString()))
                    continue;

                var createdAtRaw = element.GetProperty("created_at").GetString()!;
                var createdAt = DateTimeOffset.TryParseExact(createdAtRaw, GelbooruDateFormat, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var parsed)
                    ? parsed
                    : DateTimeOffset.UtcNow; // defensive fallback — this format is undocumented and could drift without notice

                var tags = element.GetProperty("tags").GetString()!
                    .Split(' ', StringSplitOptions.RemoveEmptyEntries)
                    .Select(t => WebUtility.HtmlDecode(t))
                    .ToArray();

                var ratingRaw = element.GetProperty("rating").GetString()!;
                posts.Add(new BooruPost(
                    element.GetProperty("id").GetInt64(),
                    md5Prop.GetString()!,
                    new Uri(fileUrlProp.GetString()!),
                    tags,
                    ratingRaw.Length > 0 ? ratingRaw[0].ToString() : ratingRaw,
                    createdAt,
                    element.GetProperty("width").GetInt32(),
                    element.GetProperty("height").GetInt32()));
            }
        }

        // Same reasoning as DanbooruClient: base "more pages exist" on the raw element
        // count, not the filtered posts list, so a filtered-out bad post can't make a
        // genuinely full page look short and stop pagination early.
        var nextCursor = rawCount == PageSize ? lastRawId!.Value.ToString(CultureInfo.InvariantCulture) : null;
        return new BooruPostPage(posts, nextCursor);
    }

    /// <summary>Gelbooru's dapi supports the same <c>id:N</c> meta-tag syntax as Danbooru, so a single-post lookup is just a one-result tag search — no separate endpoint needed.</summary>
    public async Task<BooruPost?> GetPostAsync(long postId, CancellationToken cancellationToken = default)
    {
        var page = await ListPostsAsync($"id:{postId}", cursor: null, cancellationToken).ConfigureAwait(false);
        return page.Posts.Count > 0 ? page.Posts[0] : null;
    }
}
