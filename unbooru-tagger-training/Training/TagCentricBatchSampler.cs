namespace UnbooruTagger.Training.Training;

/// <summary>One image in a <see cref="TagCentricBatchSampler"/> batch, and why it's there.</summary>
/// <param name="Image">Index into the image list <see cref="TagStatistics"/> was built from.</param>
/// <param name="AnchorTag">The tag this image was drawn for.</param>
/// <param name="IsHardNegative">True if drawn as a hard negative for <paramref name="AnchorTag"/> (trained as a certain "no").</param>
/// <param name="IsContextNegative">True if drawn as a context negative for it (<see cref="TagStatistics.ContextSources"/>; trained as an unverified "no"). Neither flag = a positive of the anchor.</param>
public readonly record struct SampledImage(int Image, int AnchorTag, bool IsHardNegative, bool IsContextNegative = false);

/// <summary>
/// Builds every batch around a handful of "anchor" tags instead of around random images:
/// for each anchor, a few of its positives AND a few of its hard negatives (images with a
/// commonly co-occurring tag but not the anchor — see <see cref="TagStatistics.HardNegativeSources"/>).
///
/// Why: with random images, a given tag almost never has a positive in a 20-image batch,
/// and the confusable negatives the crawler collected for it only land in a batch by
/// chance, so most steps teach most tags nothing beyond "say no". Pairing each anchor's
/// positives with its hard negatives in the same batch puts the exact distinction the tag
/// has to learn (this character vs. the same series' other characters, large_breasts vs.
/// breasts) in front of the loss every step. Anchors are drawn uniformly over tags with
/// enough training images, so a tag's share of the gradient doesn't depend on how common
/// it is — the long-tail balance <see cref="RareTagOversamplingBatchSampler"/> was
/// approximating with per-image weights.
/// </summary>
public sealed class TagCentricBatchSampler
{
    private readonly TagStatistics _statistics;
    private readonly int[] _anchorTags;
    private readonly int _positivesPerAnchor;
    private readonly int _hardNegativesPerAnchor;
    private readonly int _contextNegativesPerAnchor;
    private readonly Random _random;

    public TagCentricBatchSampler(
        TagStatistics statistics,
        int positivesPerAnchor,
        int hardNegativesPerAnchor,
        int minAnchorPositives = 1,
        IReadOnlySet<int>? excludedAnchors = null,
        Random? random = null,
        int contextNegativesPerAnchor = 0)
    {
        if (positivesPerAnchor < 1)
            throw new ArgumentException("positivesPerAnchor must be at least 1.", nameof(positivesPerAnchor));
        if (hardNegativesPerAnchor < 0)
            throw new ArgumentException("hardNegativesPerAnchor can't be negative.", nameof(hardNegativesPerAnchor));

        _statistics = statistics;
        _positivesPerAnchor = positivesPerAnchor;
        _hardNegativesPerAnchor = hardNegativesPerAnchor;
        _contextNegativesPerAnchor = contextNegativesPerAnchor;
        _random = random ?? new Random();
        _anchorTags = Enumerable.Range(0, statistics.VocabularySize)
            .Where(t => statistics.Positives[t].Length >= minAnchorPositives && excludedAnchors?.Contains(t) != true)
            .ToArray();

        if (_anchorTags.Length == 0)
            throw new InvalidOperationException($"No tag has at least {minAnchorPositives} training images to anchor a batch on.");
    }

    public int AnchorTagCount => _anchorTags.Length;

    /// <summary>
    /// Tags to draw <see cref="FocusShare"/> of the anchors from (<see cref="FocusTags"/>), each with
    /// as many context negatives as positives. Replaced after each epoch evaluation; tags with no
    /// training image are skipped.
    /// </summary>
    public int[] Focus
    {
        get => _focus;
        set => _focus = value.Where(t => (uint)t < (uint)_statistics.VocabularySize && _statistics.Positives[t].Length > 0).ToArray();
    }

    private int[] _focus = [];

    /// <summary>Fraction of anchors drawn from <see cref="Focus"/> (0 = uniform over every tag).</summary>
    public double FocusShare { get; init; }

    public SampledImage[] SampleBatch(int batchSize)
    {
        var batch = new List<SampledImage>(batchSize);
        var focus = _focus;
        while (batch.Count < batchSize)
        {
            var focused = focus.Length > 0 && _random.NextDouble() < FocusShare;
            var anchor = focused ? focus[_random.Next(focus.Length)] : _anchorTags[_random.Next(_anchorTags.Length)];
            var positives = _statistics.Positives[anchor];

            for (var i = 0; i < _positivesPerAnchor && batch.Count < batchSize; i++)
                batch.Add(new SampledImage(positives[_random.Next(positives.Length)], anchor, IsHardNegative: false));

            // A focus anchor's context negatives come right after its positives, before the hard negatives.
            var contextNegatives = focused ? Math.Max(_contextNegativesPerAnchor, _positivesPerAnchor) : 0;
            for (var i = 0; i < contextNegatives && batch.Count < batchSize; i++)
                if (_statistics.SampleContextNegative(anchor, _random) is int image)
                    batch.Add(new SampledImage(image, anchor, IsHardNegative: false, IsContextNegative: true));

            for (var i = 0; i < _hardNegativesPerAnchor && batch.Count < batchSize; i++)
            {
                var negative = _statistics.SampleHardNegative(anchor, _random);
                batch.Add(negative is int image
                    ? new SampledImage(image, anchor, IsHardNegative: true)
                    : new SampledImage(positives[_random.Next(positives.Length)], anchor, IsHardNegative: false));
            }

            // Only tags that have context sources get these (most tags have none).
            for (var i = 0; !focused && i < _contextNegativesPerAnchor && batch.Count < batchSize; i++)
                if (_statistics.SampleContextNegative(anchor, _random) is int image)
                    batch.Add(new SampledImage(image, anchor, IsHardNegative: false, IsContextNegative: true));
        }

        return batch.ToArray();
    }
}
