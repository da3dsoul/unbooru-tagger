namespace UnbooruTagger.Crawler;

/// <summary>
/// Pure decision logic for the crawl loop's quota/dedup/negative-shortfall math — kept
/// free of SQLite/HTTP so it's independently unit-testable, same rationale as
/// <see cref="TagEligibility"/>.
/// </summary>
public static class CrawlQuota
{
    /// <summary>
    /// Whether a tag's combined-across-both-sites count is still short of
    /// <paramref name="maxImages"/> — false once the quota is already met, at which
    /// point the crawl loop moves on to the next tag without spending any more requests
    /// on this one.
    /// </summary>
    public static bool ShouldContinueFetching(int combinedCountSoFar, int maxImages) =>
        combinedCountSoFar < maxImages;

    /// <summary>
    /// Whether a post actually needs its image bytes downloaded. Both Danbooru and
    /// Gelbooru return a post's full tag list in the same response used to find it, so
    /// a post whose md5 is already known locally never needs re-downloading or any
    /// per-tag merge — it was already recorded with every eligible tag it has the first
    /// time it was seen.
    /// </summary>
    public static bool NeedsDownload(bool md5AlreadyKnown) => !md5AlreadyKnown;

    /// <summary>
    /// How many more non-tagged images a tag still needs to reach <paramref name="negativeTarget"/>,
    /// computed as pure arithmetic against counters the crawl already maintains — no
    /// need to scan the cache or keep a separate per-image tag-membership table.
    /// Returns 0 (never negative) once the target is already met.
    /// </summary>
    public static int NegativeShortfall(int totalImages, int combinedPositiveCount, int negativeTarget)
    {
        var currentNegatives = totalImages - combinedPositiveCount;
        return Math.Max(0, negativeTarget - currentNegatives);
    }
}

/// <summary>
/// Picks which eligible tag the positive-crawl phase should work on next, and which
/// site a tag's next batch of requests should go to.
/// </summary>
public static class CrawlScheduling
{
    /// <summary>
    /// Rarest eligible tag first: guarantees rare-but-eligible tags get their full quota
    /// even if a run is interrupted before reaching common tags — same rationale as
    /// <c>TagCoverageSampleSelector</c>'s rarest-first ordering in <c>unbooru-tagger-data</c>.
    /// Ties broken by name so ordering (and therefore resumability) is deterministic.
    /// </summary>
    public static IEnumerable<TagSurveyResult> RarestFirst(IEnumerable<TagSurveyResult> eligibleTags) =>
        eligibleTags.OrderBy(t => t.BestCount).ThenBy(t => t.Name, StringComparer.Ordinal);

    /// <summary>
    /// Picks the site with fewest requests made so far this run, so load naturally
    /// spreads across both sites instead of exhausting one before the other is ever
    /// touched. Ties broken by name for determinism.
    /// </summary>
    public static string PickLeastLoadedSite(IReadOnlyDictionary<string, int> requestsMadeBySite)
    {
        if (requestsMadeBySite.Count == 0)
            throw new ArgumentException("At least one site is required.", nameof(requestsMadeBySite));

        return requestsMadeBySite
            .OrderBy(kv => kv.Value)
            .ThenBy(kv => kv.Key, StringComparer.Ordinal)
            .First()
            .Key;
    }
}
