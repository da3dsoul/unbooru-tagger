using System.Text.Json;

namespace UnbooruTagger.Crawler;

/// <summary>
/// Persists the combined Danbooru+Gelbooru active tag-alias table (antecedent raw name
/// -&gt; consequent raw name — see <see cref="DanbooruClient.ListActiveTagAliasesAsync"/>/
/// <see cref="GelbooruClient.ListActiveTagAliasesAsync"/>) to <c>--output-dir</c> so every
/// command that needs it (<see cref="TagRowMutations.BuildEligibleIdentities"/>,
/// <see cref="TagSurveyor.SurveyAsync"/>) isn't forced to re-fetch tens of thousands of
/// aliases on every single invocation — that used to happen even for a <c>crawl</c> run
/// with nothing to do with aliases, and crashed the whole command outright if a site's
/// alias listing was ever briefly unreachable.
///
/// <c>survey-tags</c> and <c>refresh-tags</c> are this cache's only writers — both
/// unconditionally re-fetch and overwrite the on-disk cache every run, since correctness
/// matters more than the network round-trip for a survey or reconciliation pass.
/// <c>crawl</c> only ever reads whatever's already there (<see cref="TryLoadAsync"/>) —
/// it has no business making itself slower or less resilient just to keep alias data
/// fresh, and by the time a dataset's <c>crawl</c> run needs alias data for real,
/// <c>survey-tags</c> has already populated the cache for it.
/// </summary>
public static class TagAliasCache
{
    public const string FileName = "tag_aliases.json";

    /// <summary>
    /// Always re-fetches every configured site's current active alias table and
    /// overwrites the on-disk cache with the merge. Returns <see langword="null"/> only
    /// if NEITHER a Danbooru nor a Gelbooru client is configured this run — nothing to
    /// fetch from, nothing to cache, and callers treat that the same as "no alias data
    /// available." On an antecedent both sites claim (rare — the two alias tables mostly
    /// cover disjoint tags), Danbooru's own mapping wins, matching <see cref="TagSurveyor"/>'s
    /// existing "Danbooru is the better-curated site" tie-break for category disagreements.
    /// </summary>
    public static async Task<Dictionary<string, string>?> FetchAndCacheAsync(
        string outputDirectory,
        IReadOnlyDictionary<string, IBooruClient> clients,
        CancellationToken cancellationToken = default)
    {
        var hasDanbooru = clients.TryGetValue("danbooru", out var danbooruClient) && danbooruClient is DanbooruClient;
        var hasGelbooru = clients.TryGetValue("gelbooru", out var gelbooruClient) && gelbooruClient is GelbooruClient;
        if (!hasDanbooru && !hasGelbooru)
            return null;

        var tagAliases = new Dictionary<string, string>(StringComparer.Ordinal);

        if (hasGelbooru)
        {
            Console.WriteLine("Fetching Gelbooru's active tag aliases...");
            var gelbooruCount = 0;
            await foreach (var alias in ((GelbooruClient)gelbooruClient!).ListActiveTagAliasesAsync(cancellationToken))
            {
                tagAliases[alias.Antecedent] = alias.Consequent;
                gelbooruCount++;
            }
            Console.WriteLine($"Loaded {gelbooruCount} active Gelbooru tag alias(es).");
        }

        if (hasDanbooru)
        {
            Console.WriteLine("Fetching Danbooru's active tag aliases...");
            var danbooruCount = 0;
            await foreach (var alias in ((DanbooruClient)danbooruClient!).ListActiveTagAliasesAsync(cancellationToken))
            {
                tagAliases[alias.Antecedent] = alias.Consequent; // Danbooru wins on overlap — fetched second so it overwrites
                danbooruCount++;
            }
            Console.WriteLine($"Loaded {danbooruCount} active Danbooru tag alias(es).");
        }

        Console.WriteLine($"{tagAliases.Count} active tag alias(es) total after merging.");

        Directory.CreateDirectory(outputDirectory);
        var path = Path.Combine(outputDirectory, FileName);
        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(tagAliases), cancellationToken).ConfigureAwait(false);

        return tagAliases;
    }

    /// <summary>Reads the cache <see cref="FetchAndCacheAsync"/> last wrote — <see langword="null"/> if it doesn't exist yet, without ever fetching or writing one itself.</summary>
    public static async Task<Dictionary<string, string>?> TryLoadAsync(string outputDirectory, CancellationToken cancellationToken = default)
    {
        var path = Path.Combine(outputDirectory, FileName);
        if (!File.Exists(path))
            return null;

        var json = await File.ReadAllTextAsync(path, cancellationToken).ConfigureAwait(false);
        return JsonSerializer.Deserialize<Dictionary<string, string>>(json);
    }
}
