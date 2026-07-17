namespace UnbooruTagger.Data;

/// <summary>
/// Picks up to <paramref name="maxPerClass"/> positive IDs (or all, if fewer) and an
/// equal-sized random sample of negative IDs, so the caller ends up with a
/// class-balanced set — kept independent of EF Core/unbooru so it's easy to unit test.
/// </summary>
public static class BalancedSampleSelector
{
    public static (IReadOnlyList<int> Positives, IReadOnlyList<int> Negatives) Select(
        IReadOnlyList<int> positiveCandidateIds,
        IReadOnlyList<int> negativeCandidateIds,
        int? maxPerClass,
        Random? random = null)
    {
        random ??= Random.Shared;

        var positiveCount = maxPerClass.HasValue
            ? Math.Min(maxPerClass.Value, positiveCandidateIds.Count)
            : positiveCandidateIds.Count;

        var positives = Shuffle(positiveCandidateIds, random).Take(positiveCount).ToList();
        var negatives = Shuffle(negativeCandidateIds, random).Take(positives.Count).ToList();

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
