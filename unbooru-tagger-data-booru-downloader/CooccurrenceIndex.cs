using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Crawler;

/// <summary>
/// One qualifying candidate returned by <see cref="TagCooccurrenceIndex.FindHardNegativeSources"/>:
/// a tag that shows up commonly alongside the target AND has enough images that carry
/// it without the target to trust as a real negative source (see that method's own doc
/// comment for the exact test).
/// </summary>
public readonly record struct CooccurringTag(
    int TagRow,
    int CooccurrenceCount,
    int OtherTagImageCount,
    int CounterExampleCount,
    double Ratio);

/// <summary>
/// Per-tag and per-tag-pair image counts derived from the corpus's own already-recorded
/// tag rows — no site querying, no SQLite, pure in-memory math (same "keep decision logic
/// free of I/O so it's independently unit-testable" rationale as <see cref="CrawlQuota"/>).
///
/// Built once per <see cref="TagCooccurrenceIndex.Build"/> call from a snapshot of
/// <c>CrawlWorkingState.ImageTagRowsByCacheRow</c>'s values — every image's already
/// eligible-tag-filtered row set, so this only ever considers the (comparatively small)
/// eligible-tag universe, never full vocabulary size.
/// </summary>
public sealed class TagCooccurrenceIndex
{
    private readonly IReadOnlyDictionary<int, int> _tagImageCounts;
    private readonly IReadOnlyDictionary<int, IReadOnlyDictionary<int, int>> _neighborsByTag;

    private TagCooccurrenceIndex(
        IReadOnlyDictionary<int, int> tagImageCounts,
        IReadOnlyDictionary<int, IReadOnlyDictionary<int, int>> neighborsByTag)
    {
        _tagImageCounts = tagImageCounts;
        _neighborsByTag = neighborsByTag;
    }

    /// <summary>
    /// Single pass over every image's tag-row set: O(images x k^2) where k is that
    /// image's own eligible-tag count (typically small, tens at most) — never
    /// O(vocabulary^2), since only pairs actually observed together in some image ever
    /// get an entry.
    /// </summary>
    public static TagCooccurrenceIndex Build(IEnumerable<IReadOnlyCollection<int>> imageTagRowSets)
    {
        var tagImageCounts = new Dictionary<int, int>();
        var pairCounts = new Dictionary<(int Min, int Max), int>();

        foreach (var tagRowSet in imageTagRowSets)
        {
            if (tagRowSet.Count == 0)
                continue;

            var rows = tagRowSet as IReadOnlyList<int> ?? tagRowSet.ToArray();
            for (var i = 0; i < rows.Count; i++)
            {
                tagImageCounts[rows[i]] = tagImageCounts.GetValueOrDefault(rows[i]) + 1;

                for (var j = i + 1; j < rows.Count; j++)
                {
                    var a = rows[i];
                    var b = rows[j];
                    if (a == b)
                        continue; // a HashSet source never has this, but a caller could pass a plain list

                    var key = a < b ? (a, b) : (b, a);
                    pairCounts[key] = pairCounts.GetValueOrDefault(key) + 1;
                }
            }
        }

        var neighborsByTag = new Dictionary<int, Dictionary<int, int>>();
        foreach (var ((a, b), count) in pairCounts)
        {
            AddNeighbor(neighborsByTag, a, b, count);
            AddNeighbor(neighborsByTag, b, a, count);
        }

        return new TagCooccurrenceIndex(
            tagImageCounts,
            neighborsByTag.ToDictionary(kv => kv.Key, kv => (IReadOnlyDictionary<int, int>)kv.Value));
    }

    private static void AddNeighbor(Dictionary<int, Dictionary<int, int>> neighborsByTag, int from, int to, int count)
    {
        if (!neighborsByTag.TryGetValue(from, out var neighbors))
        {
            neighbors = [];
            neighborsByTag[from] = neighbors;
        }

        neighbors[to] = count;
    }

    /// <summary>
    /// Every other tag that (a) shows up alongside <paramref name="targetTagRow"/> in at
    /// least <paramref name="minCooccurrenceRatio"/> of the target's own images, AND
    /// (b) has at least <paramref name="minCounterExamples"/> images that carry it
    /// WITHOUT the target — the population a hard-negative query for the target would
    /// actually draw from. Requiring both is what tells apart a near-subset relationship
    /// (e.g. large_breasts implies breasts almost always: querying "breasts -large_breasts"
    /// as a negative source for large_breasts is fine since plenty of breasts images lack
    /// large_breasts, but querying "large_breasts -breasts" as a source for breasts would
    /// return almost nothing — (b) fails for large_breasts as a candidate there) from a
    /// genuinely useful pair with a real spread of counter-examples on both sides (e.g. a
    /// character and the series it's from: most art of a series isn't that one character).
    /// Ordered by <see cref="CooccurringTag.Ratio"/> descending (strongest association
    /// first), tied broken by counter-example count then by tag row for determinism.
    /// Empty if the target was never observed.
    /// </summary>
    public IReadOnlyList<CooccurringTag> FindHardNegativeSources(
        int targetTagRow, double minCooccurrenceRatio, int minCounterExamples)
    {
        if (!_tagImageCounts.TryGetValue(targetTagRow, out var targetCount)
            || !_neighborsByTag.TryGetValue(targetTagRow, out var neighbors))
        {
            return [];
        }

        var candidates = new List<CooccurringTag>();
        foreach (var (otherRow, cooccurrenceCount) in neighbors)
        {
            var ratio = (double)cooccurrenceCount / targetCount;
            if (ratio < minCooccurrenceRatio)
                continue;

            var otherCount = _tagImageCounts[otherRow];
            var counterExamples = otherCount - cooccurrenceCount;
            if (counterExamples < minCounterExamples)
                continue;

            candidates.Add(new CooccurringTag(otherRow, cooccurrenceCount, otherCount, counterExamples, ratio));
        }

        return candidates
            .OrderByDescending(c => c.Ratio)
            .ThenByDescending(c => c.CounterExampleCount)
            .ThenBy(c => c.TagRow)
            .ToList();
    }
}

/// <summary>
/// One query <see cref="NegativeQueryPlanning.BuildQuerySequence"/> wants the negative
/// phase to try for a tag, in order — either a hard-negative source (a commonly
/// co-occurring tag with enough counter-examples, see <see cref="TagCooccurrenceIndex.FindHardNegativeSources"/>)
/// or the plain tag-absent fallback every tag has always used.
/// </summary>
public readonly record struct NegativeQueryPlan(string TagQuery, string PhaseKey, string DisplayLabel);

/// <summary>
/// Turns a target tag's qualifying hard-negative sources into the ordered sequence of
/// site queries <see cref="DatasetCrawler"/>'s negative phase should try, always ending
/// with the same plain <c>-{tag}</c> query (and the same <c>"negative"</c> phase key)
/// every tag has always fallen back to — so a tag with no qualifying candidates (never
/// observed yet, or below the ratio/counter-example floors) degenerates to exactly
/// today's single-query behavior.
/// </summary>
public static class NegativeQueryPlanning
{
    public static IReadOnlyList<NegativeQueryPlan> BuildQuerySequence(
        string targetTagIdentity,
        int? targetTagRow,
        TagCooccurrenceIndex cooccurrenceIndex,
        TagVocabulary vocabulary,
        double minCooccurrenceRatio,
        int minCounterExamples,
        int maxHardNegativeSources)
    {
        var plans = new List<NegativeQueryPlan>();
        var targetRaw = TagCategoryNaming.RawName(targetTagIdentity);

        if (targetTagRow is int row && maxHardNegativeSources > 0)
        {
            var candidates = cooccurrenceIndex.FindHardNegativeSources(row, minCooccurrenceRatio, minCounterExamples);
            foreach (var candidate in candidates.Take(maxHardNegativeSources))
            {
                var candidateIdentity = vocabulary.GetByRowIndex(candidate.TagRow).Tag;
                var candidateRaw = TagCategoryNaming.RawName(candidateIdentity);
                plans.Add(new NegativeQueryPlan(
                    $"{candidateRaw} -{targetRaw}",
                    $"negative:cooccur:{candidateIdentity}",
                    candidateIdentity));
            }
        }

        plans.Add(new NegativeQueryPlan($"-{targetRaw}", "negative", "background"));
        return plans;
    }
}
