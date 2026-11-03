namespace UnbooruTagger.Training.Training;

/// <summary>
/// Context-prior debiasing (product of experts). For each training image and localizable general tag t, the
/// image's context prior is the highest P(t | s) over its character, artist and series tags s
/// (<see cref="ContextRelianceEvaluator.IsContextSource"/>), estimated from a sample of s's training images.
/// Training adds <c>weight · (logit(prior) − logit(base rate of t))</c> to the pair's whole-image logit where
/// that is positive; inference and evaluation never see it.
/// <para>
/// Where context already explains a tag ("this character nearly always wears gloves"), the loss is mostly
/// satisfied before the image tower contributes anything, so the tower gains little by learning that shortcut
/// itself; it learns gloves from the images where context doesn't predict them. A context negative (her
/// images without gloves) starts with a high logit and is pushed down hard, against the character's features.
/// The occlusion check showed the shortcut: blanking gloves lowered the score by about 1 point and left 100%
/// of calls standing.
/// </para>
/// Built from tag counts alone, so a tag added later gets its prior the same way.
/// </summary>
public sealed class ContextPrior
{
    /// <summary>Highest prior counted, so one source can't add an unbounded offset.</summary>
    public const double MaxPrior = 0.95;

    /// <summary>[source s] → (tag t, offset) for every eligible t that s raises above its base rate.</summary>
    private readonly Dictionary<int, (int Tag, float Offset)[]> _offsets;

    public float Weight { get; }
    public int SourceCount => _offsets.Count;
    public int PairCount => _offsets.Values.Sum(o => o.Length);

    private ContextPrior(Dictionary<int, (int, float)[]> offsets, float weight)
    {
        _offsets = offsets;
        Weight = weight;
    }

    /// <param name="isSource">Per tag row: a context source (character/artist/series).</param>
    /// <param name="eligible">Per tag row: offset this tag (training passes localizable general tags).</param>
    /// <param name="minSourceImages">Sources with fewer training images are skipped: too few to estimate a prior.</param>
    /// <param name="minPrior">Priors below this are left out.</param>
    /// <param name="minCount">Sampled co-occurrences needed for a prior to count at all.</param>
    public static ContextPrior Build(
        TagStatistics statistics, IReadOnlyList<bool> isSource, IReadOnlyList<bool> eligible, float weight,
        int samplesPerSource = 512, int minSourceImages = 50, double minPrior = 0.1, int minCount = 5, int seed = 0)
    {
        var imageCount = Math.Max(1, statistics.ImageTagRows.Count);
        var offsets = new Dictionary<int, (int, float)[]>();
        var lockObject = new object();
        Parallel.For(0, statistics.VocabularySize, s =>
        {
            if (s >= isSource.Count || !isSource[s] || statistics.Positives[s].Length < minSourceImages)
                return;
            var images = statistics.Positives[s];
            var random = new Random(DeterministicSeed.For(seed, s));
            var sample = images.Length <= samplesPerSource ? images : images.OrderBy(_ => random.Next()).Take(samplesPerSource).ToArray();
            var tally = new Dictionary<int, int>();
            foreach (var image in sample)
                foreach (var t in statistics.ImageTagRows[image])
                    if (t != s && (uint)t < (uint)eligible.Count && eligible[t])
                        tally[t] = tally.GetValueOrDefault(t) + 1;

            var raised = new List<(int, float)>();
            foreach (var (t, count) in tally)
            {
                var prior = count / (double)sample.Length;
                if (count < minCount || prior < minPrior)
                    continue;
                var baseRate = Math.Max(statistics.Positives[t].Length / (double)imageCount, 1e-5);
                var offset = weight * (Logit(Math.Min(prior, MaxPrior)) - Logit(Math.Min(baseRate, MaxPrior)));
                if (offset > 0)
                    raised.Add((t, (float)offset));
            }

            if (raised.Count > 0)
                lock (lockObject)
                    offsets[s] = raised.ToArray();
        });
        return new ContextPrior(offsets, weight);
    }

    /// <summary>Writes one image's offsets (the highest over its sources, per tag) into <paramref name="row"/>, a [vocabulary] slice.</summary>
    public void Fill(IReadOnlyList<int> imageTags, Span<float> row)
    {
        foreach (var s in imageTags)
            if (_offsets.TryGetValue(s, out var raised))
                foreach (var (t, offset) in raised)
                    if (offset > row[t])
                        row[t] = offset;
    }

    private static double Logit(double p) => Math.Log(p / (1 - p));
}

/// <summary>
/// Each image's context prior for a tag: the highest P(t | s) over its character, artist and series tags s,
/// estimated from a sample of s's training images (as <see cref="ContextPrior"/> and
/// <see cref="ContextRelianceEvaluator"/> do). <see cref="TagCentricBatchSampler"/> uses it to pick
/// out-of-context positives: gloves on a character who rarely wears them, which can only be learned from the
/// gloves themselves.
/// </summary>
public sealed class ContextPriorTable
{
    /// <summary>[source s] → P(t | s) for every eligible t seen in its sample.</summary>
    private readonly Dictionary<int, Dictionary<int, float>> _priors;

    public int SourceCount => _priors.Count;

    private ContextPriorTable(Dictionary<int, Dictionary<int, float>> priors) => _priors = priors;

    /// <param name="isSource">Per tag row: a context source (character/artist/series).</param>
    /// <param name="eligible">Per tag row: record priors for this tag.</param>
    /// <param name="minSourceImages">Sources with fewer training images are skipped: too few to estimate a prior.</param>
    public static ContextPriorTable Build(
        TagStatistics statistics, IReadOnlyList<bool> isSource, IReadOnlyList<bool> eligible,
        int samplesPerSource = 512, int minSourceImages = 20, int seed = 0)
    {
        var priors = new Dictionary<int, Dictionary<int, float>>();
        var lockObject = new object();
        Parallel.For(0, statistics.VocabularySize, s =>
        {
            if (s >= isSource.Count || !isSource[s] || statistics.Positives[s].Length < minSourceImages)
                return;
            var images = statistics.Positives[s];
            var random = new Random(DeterministicSeed.For(seed, s));
            var sample = images.Length <= samplesPerSource ? images : images.OrderBy(_ => random.Next()).Take(samplesPerSource).ToArray();
            var tally = new Dictionary<int, int>();
            foreach (var image in sample)
                foreach (var t in statistics.ImageTagRows[image])
                    if (t != s && (uint)t < (uint)eligible.Count && eligible[t])
                        tally[t] = tally.GetValueOrDefault(t) + 1;
            var row = tally.ToDictionary(kv => kv.Key, kv => kv.Value / (float)sample.Length);
            lock (lockObject)
                priors[s] = row;
        });
        return new ContextPriorTable(priors);
    }

    /// <summary>Whether any of <paramref name="imageTags"/> is a source with priors (so a prior of 0 means "rarely", not "unknown").</summary>
    public bool HasSource(IReadOnlyList<int> imageTags)
    {
        foreach (var s in imageTags)
            if (_priors.ContainsKey(s))
                return true;
        return false;
    }

    /// <summary>The tags some source among <paramref name="imageTags"/> gives a prior of at least <paramref name="minPrior"/> (repeats possible).</summary>
    public IEnumerable<int> LikelyTags(IReadOnlyList<int> imageTags, float minPrior)
    {
        foreach (var s in imageTags)
            if (_priors.TryGetValue(s, out var row))
                foreach (var (t, p) in row)
                    if (p >= minPrior)
                        yield return t;
    }

    /// <summary>The highest P(<paramref name="tag"/> | s) over <paramref name="imageTags"/>' sources; 0 without any.</summary>
    public float Prior(IReadOnlyList<int> imageTags, int tag)
    {
        var best = 0f;
        foreach (var s in imageTags)
            if (_priors.TryGetValue(s, out var row) && row.TryGetValue(tag, out var p) && p > best)
                best = p;
        return best;
    }
}
