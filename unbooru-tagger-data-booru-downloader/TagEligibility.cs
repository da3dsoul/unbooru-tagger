namespace UnbooruTagger.Crawler;

/// <summary>One tag's surveyed post count per site, as recorded by <c>survey-tags</c>.</summary>
public readonly record struct TagSurveyResult(string Name, int? DanbooruCount, int? GelbooruCount)
{
    /// <summary>The count a crawl budgets against — the larger of the two sites' counts, since a tag only needs to be worth crawling on at least one of them.</summary>
    public int BestCount => Math.Max(DanbooruCount ?? 0, GelbooruCount ?? 0);
}

/// <summary>
/// Pure tag-eligibility/estimation logic, kept independent of SQLite/HTTP so it's
/// trivially unit-testable — mirrors how <c>unbooru-tagger-data</c> keeps its sample
/// selectors (<c>BalancedSampleSelector</c>, <c>TagCoverageSampleSelector</c>) free of
/// EF Core.
/// </summary>
public static class TagEligibility
{
    /// <summary>A tag is worth crawling once at least one site's post count clears <paramref name="minImages"/>.</summary>
    public static bool IsEligible(TagSurveyResult tag, int minImages) => tag.BestCount >= minImages;

    /// <summary>
    /// Pre-dedup upper bound on total image slots a crawl would need: summing
    /// <c>min(maxImages, count)</c> across every eligible tag. This can only ever be an
    /// upper bound — real crawls end up lower once cross-tag/cross-site dedup kicks in,
    /// which isn't knowable until the crawl actually runs.
    /// </summary>
    public static long EstimateImageSlots(IEnumerable<TagSurveyResult> tags, int minImages, int maxImages) =>
        tags.Where(t => IsEligible(t, minImages)).Sum(t => (long)Math.Min(maxImages, t.BestCount));
}
