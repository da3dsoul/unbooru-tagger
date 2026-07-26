using System.Text.Json;

namespace UnbooruTagger.Crawler;

/// <summary>
/// Persists Danbooru's active tag-alias table (antecedent raw name -&gt; consequent raw
/// name — see <see cref="DanbooruClient.ListActiveTagAliasesAsync"/>) to <c>--output-dir</c>
/// so every command that needs it (<see cref="TagRowMutations.BuildEligibleIdentities"/>,
/// <see cref="TagSurveyor.SurveyAsync"/>) isn't forced to re-fetch tens of thousands of
/// aliases from Danbooru on every single invocation — that used to happen even for a
/// <c>crawl</c> run with nothing to do with aliases, and crashed the whole command outright
/// if Danbooru's alias endpoint was ever briefly unreachable.
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
    /// Always re-fetches Danbooru's current active alias table and overwrites the
    /// on-disk cache with it. Returns <see langword="null"/> if no Danbooru client is
    /// configured this run — nothing to fetch from, nothing to cache, and callers treat
    /// that the same as "no alias data available."
    /// </summary>
    public static async Task<Dictionary<string, string>?> FetchAndCacheAsync(
        string outputDirectory,
        IReadOnlyDictionary<string, IBooruClient> clients,
        CancellationToken cancellationToken = default)
    {
        if (!clients.TryGetValue("danbooru", out var danbooruClient) || danbooruClient is not DanbooruClient danbooru)
            return null;

        Console.WriteLine("Fetching Danbooru's active tag aliases...");
        var tagAliases = new Dictionary<string, string>(StringComparer.Ordinal);
        await foreach (var alias in danbooru.ListActiveTagAliasesAsync(cancellationToken))
            tagAliases[alias.Antecedent] = alias.Consequent;
        Console.WriteLine($"Loaded {tagAliases.Count} active tag alias(es).");

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
