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

        // Seeded first (so a later official alias for the same antecedent overwrites it)
        // from the redirects 'detect-deprecated-redirects' empirically confirmed. Those
        // aren't re-derivable from any site's API — they exist precisely because the site
        // publishes no trace of them — so an unconditional overwrite of this file with
        // freshly-fetched official aliases, which is what every run of this method does,
        // would otherwise silently discard hours of rate-limited detection work the first
        // time anyone re-ran survey-tags.
        var detectedRedirects = await TryLoadDetectedRedirectsAsync(outputDirectory, cancellationToken).ConfigureAwait(false);
        if (detectedRedirects is { Count: > 0 })
        {
            foreach (var (antecedent, consequent) in detectedRedirects)
                tagAliases[antecedent] = consequent;
            Console.WriteLine($"Loaded {detectedRedirects.Count} previously-detected silent redirect(s) from '{DetectedRedirectsFileName}'.");
        }

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

    /// <summary>
    /// A durable, append-only-in-practice record of the silent search redirects
    /// <see cref="DeprecatedRedirectDetector"/> confirmed empirically (same flat
    /// antecedent -&gt; consequent shape as <see cref="FileName"/>). Kept as its own file
    /// purely so <see cref="FetchAndCacheAsync"/> — which overwrites <see cref="FileName"/>
    /// outright on every survey-tags/refresh-tags run — can fold them back in rather than
    /// destroy them; <see cref="FileName"/> stays the single file every other reader
    /// (<c>crawl</c>, <see cref="TagRowMutations.BuildEligibleIdentities"/>, ...) consults.
    /// </summary>
    public const string DetectedRedirectsFileName = "detected_tag_redirects.json";

    /// <summary>Reads <see cref="DetectedRedirectsFileName"/> — <see langword="null"/> if no detection run has ever written one under this directory.</summary>
    public static async Task<Dictionary<string, string>?> TryLoadDetectedRedirectsAsync(string outputDirectory, CancellationToken cancellationToken = default)
    {
        var path = Path.Combine(outputDirectory, DetectedRedirectsFileName);
        if (!File.Exists(path))
            return null;

        var json = await File.ReadAllTextAsync(path, cancellationToken).ConfigureAwait(false);
        return JsonSerializer.Deserialize<Dictionary<string, string>>(json);
    }

    /// <summary>
    /// Merges <paramref name="additions"/> into the on-disk <see cref="FileName"/> cache and
    /// rewrites it, returning only the entries that were genuinely new. Deliberately a
    /// merge, never a replace: this file is the combined official Danbooru+Gelbooru alias
    /// table that <c>crawl</c>/<see cref="TagRowMutations.BuildEligibleIdentities"/> read,
    /// and a caller with a handful of extra mappings has no business dropping tens of
    /// thousands of existing ones. An antecedent already present is left exactly as it is
    /// rather than overwritten — an official alias a site actually publishes outranks
    /// anything inferred empirically, and silently retargeting one would be the worst
    /// possible way to find out the two disagreed.
    /// </summary>
    public static async Task<IReadOnlyDictionary<string, string>> MergeIntoCacheAsync(
        string outputDirectory,
        IReadOnlyDictionary<string, string> additions,
        CancellationToken cancellationToken = default)
    {
        var merged = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var (antecedent, consequent) in await TryLoadAsync(outputDirectory, cancellationToken).ConfigureAwait(false) ?? [])
            merged[antecedent] = consequent;

        var added = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var (antecedent, consequent) in additions)
        {
            if (merged.ContainsKey(antecedent))
                continue;
            merged[antecedent] = consequent;
            added[antecedent] = consequent;
        }

        Directory.CreateDirectory(outputDirectory);
        await File.WriteAllTextAsync(Path.Combine(outputDirectory, FileName), JsonSerializer.Serialize(merged), cancellationToken).ConfigureAwait(false);
        return added;
    }

    /// <summary>Overwrites <see cref="DetectedRedirectsFileName"/> with <paramref name="detectedRedirects"/> — callers are expected to have merged whatever <see cref="TryLoadDetectedRedirectsAsync"/> returned into it first, since a detection run only ever probes the tags it hasn't already resolved.</summary>
    public static async Task SaveDetectedRedirectsAsync(string outputDirectory, IReadOnlyDictionary<string, string> detectedRedirects, CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(outputDirectory);
        var path = Path.Combine(outputDirectory, DetectedRedirectsFileName);
        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(detectedRedirects), cancellationToken).ConfigureAwait(false);
    }
}
