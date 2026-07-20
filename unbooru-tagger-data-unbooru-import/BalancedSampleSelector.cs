namespace UnbooruTagger.Data;

/// <summary>
/// Picks up to <paramref name="maxPerClass"/> positives (or all, if fewer) from the
/// full matching pool. Negatives are then drawn from a mix of whatever matching images
/// were left over (not selected as positives — these still share 1-to-all of the
/// target tags with the positive set) and the true zero-overlap candidates, so the
/// negative set spans a spread of tag-overlap counts rather than uniformly "none of the
/// target tags". Falls back to zero-overlap-only negatives when there's no leftover
/// (e.g. every matching image was already used as a positive) — kept independent of EF
/// Core/unbooru so it's easy to unit test.
/// </summary>
public static class BalancedSampleSelector
{
    public static (IReadOnlyList<int> Positives, IReadOnlyList<int> Negatives) Select(
        IReadOnlyList<int> matchingCandidateIds,
        IReadOnlyList<int> zeroOverlapCandidateIds,
        int? maxPerClass,
        Random? random = null)
    {
        random ??= Random.Shared;

        var shuffledMatching = Shuffle(matchingCandidateIds, random);
        var positiveCount = maxPerClass.HasValue
            ? Math.Min(maxPerClass.Value, shuffledMatching.Count)
            : shuffledMatching.Count;

        var positives = shuffledMatching.Take(positiveCount).ToList();
        var leftoverMatching = shuffledMatching.Skip(positiveCount).ToList();

        var negativePool = leftoverMatching.Concat(zeroOverlapCandidateIds).ToList();
        var negatives = Shuffle(negativePool, random).Take(positives.Count).ToList();

        return (positives, negatives);
    }

    private static List<int> Shuffle(IReadOnlyList<int> source, Random random)
    {
        var items = source.ToList();
        for (var i = items.Count - 1; i > 0; i--)
        {
            var j = random.Next(i + 1);
            (items[i], items[j]) = (items[j], items[i]);
        }
        return items;
    }
}
