using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text.Json;

namespace UnbooruTagger.Crawler;

/// <summary>
/// Thin Danbooru API client. Authenticates via the officially documented
/// <c>login</c>+<c>api_key</c> query params — deliberately not the session-cookie
/// (<c>user_id</c>/<c>pass_hash</c>) approach <c>BooruSharp</c>'s Danbooru template
/// uses, which is very likely why <c>Xwilarg/BooruSharp#53</c> ("authentication fails
/// regardless of credentials") is open. Read <c>rating</c> as its first character only
/// (<c>value[0]</c>), not a fixed enum string — Danbooru has changed rating
/// representation before (single-letter historically, full words today; both forms
/// share unique first letters), so this is a cheap forward/backward-compatible read.
/// </summary>
public sealed class DanbooruClient(HttpClient http, IRateLimiter rateLimiter, string? login = null, string? apiKey = null, string baseUrl = "https://danbooru.donmai.us")
    : BooruHttpClientBase(http, rateLimiter), IBooruClient
{
    private readonly string _baseUrl = baseUrl.TrimEnd('/');

    public string SiteName => "danbooru";

    /// <summary>Danbooru's documented per-page maximum for <c>posts.json</c>.</summary>
    public int PageSize => 200;

    private string AuthQuery() =>
        login is not null && apiKey is not null
            ? $"&login={Uri.EscapeDataString(login)}&api_key={Uri.EscapeDataString(apiKey)}"
            : "";

    public async IAsyncEnumerable<BooruTagCount> ListTagsByCountDescendingAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        for (var page = 1; ; page++)
        {
            var uri = new Uri($"{_baseUrl}/tags.json?search[order]=count&search[hide_empty]=true&limit=1000&page={page}{AuthQuery()}");
            var json = await GetStringAsync(uri, cancellationToken).ConfigureAwait(false);
            using var doc = JsonDocument.Parse(json);

            var any = false;
            foreach (var element in doc.RootElement.EnumerateArray())
            {
                any = true;
                yield return new BooruTagCount(
                    element.GetProperty("name").GetString()!,
                    element.GetProperty("post_count").GetInt32());
            }

            if (!any)
                yield break;
        }
    }

    public async Task<BooruPostPage> ListPostsAsync(string tagQuery, string? cursor, CancellationToken cancellationToken = default)
    {
        var cursorQuery = cursor is null ? "" : $"&page=b{cursor}";
        var uri = new Uri($"{_baseUrl}/posts.json?tags={Uri.EscapeDataString(tagQuery)}&limit={PageSize}{cursorQuery}{AuthQuery()}");
        var json = await GetStringAsync(uri, cancellationToken).ConfigureAwait(false);
        using var doc = JsonDocument.Parse(json);

        var posts = new List<BooruPost>();
        var rawCount = 0;
        long? lastRawId = null;
        foreach (var element in doc.RootElement.EnumerateArray())
        {
            rawCount++;
            var id = element.GetProperty("id").GetInt64();
            lastRawId = id;

            // Deleted/flagged/banned posts can be missing md5 or file_url entirely —
            // skip rather than throw, since one bad post shouldn't kill an entire page.
            if (!element.TryGetProperty("md5", out var md5Prop) || md5Prop.ValueKind != JsonValueKind.String)
                continue;
            if (!element.TryGetProperty("file_url", out var fileUrlProp) || fileUrlProp.ValueKind != JsonValueKind.String)
                continue;

            var ratingRaw = element.GetProperty("rating").GetString()!;
            posts.Add(new BooruPost(
                id,
                md5Prop.GetString()!,
                new Uri(fileUrlProp.GetString()!),
                element.GetProperty("tag_string").GetString()!.Split(' ', StringSplitOptions.RemoveEmptyEntries),
                ratingRaw.Length > 0 ? ratingRaw[0].ToString() : ratingRaw,
                element.GetProperty("created_at").GetDateTimeOffset(),
                element.GetProperty("image_width").GetInt32(),
                element.GetProperty("image_height").GetInt32()));
        }

        // A short raw page (fewer elements than PageSize, counted before filtering out
        // any bad posts) means this tag query is exhausted — Danbooru's page=bN cursor
        // pagination has nothing further past this point. Basing this on the raw count
        // rather than posts.Count matters because a page that happens to contain a
        // filtered-out deleted post would otherwise look short and stop paging early.
        var nextCursor = rawCount == PageSize ? lastRawId!.Value.ToString(CultureInfo.InvariantCulture) : null;
        return new BooruPostPage(posts, nextCursor);
    }

    /// <summary><c>id:N</c> is a documented Danbooru meta-tag, so a single-post lookup is just a one-result tag search — no separate endpoint needed.</summary>
    public async Task<BooruPost?> GetPostAsync(long postId, CancellationToken cancellationToken = default)
    {
        var page = await ListPostsAsync($"id:{postId}", cursor: null, cancellationToken).ConfigureAwait(false);
        return page.Posts.Count > 0 ? page.Posts[0] : null;
    }
}
