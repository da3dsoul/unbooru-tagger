namespace UnbooruTagger.Training.Training;

/// <summary>
/// Weights image sampling by the rarity of its rarest tag, so common tags (1girl,
/// solo, ...) don't dominate gradients and starve the long tail — CLAUDE.md calls this
/// out as mattering "more than almost anything else" at this vocabulary scale.
/// </summary>
public sealed class RareTagOversamplingBatchSampler
{
    private readonly double[] _imageWeights;
    private readonly double _totalWeight;
    private readonly Random _random;

    public RareTagOversamplingBatchSampler(
        IReadOnlyList<IReadOnlyList<int>> imageTagRows,
        IReadOnlyDictionary<int, int> tagFrequencies,
        Random? random = null)
    {
        _random = random ?? new Random();
        _imageWeights = imageTagRows
            .Select(tags => tags.Count == 0 ? 0d : 1d / tags.Min(t => tagFrequencies.GetValueOrDefault(t, 1)))
            .ToArray();
        _totalWeight = _imageWeights.Sum();
    }

    public int[] SampleBatch(int batchSize)
    {
        var batch = new int[batchSize];
        for (var i = 0; i < batchSize; i++)
            batch[i] = SampleOne();
        return batch;
    }

    private int SampleOne()
    {
        var target = _random.NextDouble() * _totalWeight;
        var cumulative = 0d;
        for (var i = 0; i < _imageWeights.Length; i++)
        {
            cumulative += _imageWeights[i];
            if (cumulative >= target)
                return i;
        }
        return _imageWeights.Length - 1;
    }
}
