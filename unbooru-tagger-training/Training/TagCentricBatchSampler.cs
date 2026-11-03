namespace UnbooruTagger.Training.Training;

/// <summary>One image in a <see cref="TagCentricBatchSampler"/> batch, and why it's there.</summary>
/// <param name="Image">Index into the image list <see cref="TagStatistics"/> was built from.</param>
/// <param name="AnchorTag">The tag this image was drawn for.</param>
/// <param name="IsHardNegative">True if drawn as a hard negative for <paramref name="AnchorTag"/> (trained as a certain "no").</param>
/// <param name="IsContextNegative">True if drawn as a context negative for it (<see cref="TagStatistics.ContextSources"/>; trained as an unverified "no"). Neither flag = a positive of the anchor.</param>
/// <param name="IsOutOfContextPositive">A positive of the anchor drawn for its low context prior (<see cref="TagCentricBatchSampler.OutOfContextPriors"/>).</param>
/// <param name="ContextPair">Above -1, this image is half of a context pair (<see cref="TagCentricBatchSampler.ContextPairsPerAnchor"/>): the
/// batch's two images with this number share one of the anchor's context sources, one with the anchor (a positive) and one
/// without it (<paramref name="IsContextNegative"/>).</param>
public readonly record struct SampledImage(int Image, int AnchorTag, bool IsHardNegative, bool IsContextNegative = false, bool IsOutOfContextPositive = false, int ContextPair = -1, bool IsTwin = false);

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

    /// <summary>
    /// Each batch starts with this many context pairs (<see cref="TagStatistics.SampleContextPair"/>): two images of one source
    /// (a character, artist or series), one with an anchor tag and one without, the anchor drawn uniformly from
    /// <see cref="ContextPairTags"/>. Whatever identifies the source is in both, so only something else can separate them
    /// (2026-11-02: a free glimpse tower under a sharp pool had learned the image's identity in every window). Drawn per anchor
    /// instead, a pair fitted in a batch only when its first anchor had context sources (about one tag in eight). 0 = off.
    /// </summary>
    public int ContextPairsPerBatch { get; init; }

    /// <summary>The tags <see cref="ContextPairsPerBatch"/> draws its anchors from (null = every tag with context sources).</summary>
    public IReadOnlyList<int>? ContextPairTags { get; set; }

    /// <summary>
    /// Images at the start of each batch for the glimpse feature split's exclusivity slice (<c>--glimpse-exclusive-per-batch</c>):
    /// alternately a context negative (a context source without the tag: the character without her gloves) and an exclusive
    /// positive (<see cref="ExclusivePositive"/>), each anchored on a tag drawn uniformly from <see cref="ContextPairTags"/>
    /// that has context sources. A share of the batch instead of a heavier weight. 0 = off.
    /// </summary>
    public int ExclusivePerBatch { get; init; }

    /// <summary>The feature split's priors (character/artist/series), for <see cref="ExclusivePerBatch"/>'s exclusive positives.</summary>
    public ContextPriorTable? ExclusivePriors { get; init; }

    /// <summary>An exclusive positive's highest prior: below it, the image's characters/artists/series rarely carry the tag.</summary>
    public float ExclusiveMaxPrior { get; init; } = 0.2f;

    /// <summary>
    /// A positive of <paramref name="tag"/> whose image has a known character/artist/series that rarely carries it
    /// (<see cref="ExclusivePriors"/> below <see cref="ExclusiveMaxPrior"/>; Singh et al. 2020's exclusive image, where identity
    /// says "no" and only the object can carry the label); null if none is found in <paramref name="attempts"/> tries.
    /// </summary>
    public int? ExclusivePositive(int tag, int attempts = 64)
    {
        var positives = _statistics.Positives[tag];
        if (ExclusivePriors is not { } priors || positives.Length == 0)
            return null;
        for (var attempt = 0; attempt < attempts; attempt++)
        {
            var image = positives[_random.Next(positives.Length)];
            var tags = _statistics.ImageTagRows[image];
            if (priors.HasSource(tags) && priors.Prior(tags, tag) < ExclusiveMaxPrior)
                return image;
        }
        return null;
    }

    private int[]? _contextPairTags;

    /// <summary>
    /// Out-of-context positives: with a table, <see cref="OutOfContextShare"/> of an eligible anchor's positives are
    /// the lowest-prior of <see cref="OutOfContextCandidates"/> random ones, kept as such if that prior is below
    /// <see cref="OutOfContextMaxPrior"/> (gloves on a character who rarely wears them). Null = off.
    /// </summary>
    public ContextPriorTable? OutOfContextPriors { get; init; }

    /// <summary>Per tag row: may get out-of-context positives (training passes localizable general tags).</summary>
    public IReadOnlyList<bool>? OutOfContextTags { get; init; }

    public double OutOfContextShare { get; init; }
    public double OutOfContextMaxPrior { get; init; } = 0.3;
    public int OutOfContextCandidates { get; init; } = 16;

    /// <summary>
    /// Fraction of anchors drawn from the tags an excludes rule rules out (<see cref="TagFamilies.Excluders"/>: gloves,
    /// for no_gloves; parents only, uniformly), each with two of its excluders' images as certain negatives: the character usually has gloves,
    /// this image says she doesn't. Those images are rare (no_gloves: ~1K of 5M), so without this they reach
    /// training only by chance. 0 = off.
    /// </summary>
    public double ExcluderShare { get; init; }

    private int[]? _excludedTags;

    private int? SampleExcluderNegative(int anchor)
    {
        var excluders = _statistics.Families.Excluders(anchor);
        for (var attempt = 0; attempt < 16 && excluders.Count > 0; attempt++)
        {
            var images = _statistics.Positives[excluders[_random.Next(excluders.Count)]];
            if (images.Length == 0)
                continue;
            var image = images[_random.Next(images.Length)];
            // Only where the rule applies: no_shoes on an image also tagged shoes (a second character's) rules nothing out.
            if (_statistics.Families.ExcludedRows(_statistics.ImageTagRows[image]).Contains(anchor))
                return image;
        }
        return null;
    }

    private SampledImage SamplePositive(int anchor, int[] positives)
    {
        if (OutOfContextPriors is { } priors && OutOfContextTags is { } tags && anchor < tags.Count && tags[anchor] && _random.NextDouble() < OutOfContextShare)
        {
            var (best, bestPrior) = (-1, float.MaxValue);
            for (var i = 0; i < OutOfContextCandidates; i++)
            {
                var candidate = positives[_random.Next(positives.Length)];
                var prior = priors.Prior(_statistics.ImageTagRows[candidate], anchor);
                if (prior < bestPrior)
                    (best, bestPrior) = (candidate, prior);
            }
            if (bestPrior < OutOfContextMaxPrior)
                return new SampledImage(best, anchor, IsHardNegative: false, IsOutOfContextPositive: true);
        }
        return new SampledImage(positives[_random.Next(positives.Length)], anchor, IsHardNegative: false);
    }

    /// <summary>
    /// Matched twins (the user's idea, 2026-11-02): every <see cref="TwinEvery"/>th batch is one image with a
    /// <see cref="TwinTags"/> tag and as many images as possible without it whose tags otherwise overlap it most (Jaccard over the
    /// tag sets), each with at most one character. What differs between the one and its twins is mostly the tag: a window of the
    /// positive that looks like the twins' windows is a window without it (<see cref="SampledImage.IsTwin"/>; the training step's
    /// twin term; the positive comes first, and the twins are ordinary negatives, not certain ones: ~3% of a tag's negatives
    /// are missing labels). 0 = off.
    /// </summary>
    public int TwinEvery { get; init; }

    /// <summary>The tags twin batches are anchored on (localizable general tags).</summary>
    public IReadOnlyList<int>? TwinTags { get; init; }

    /// <summary>Which rows are characters (at most one per twin-batch image).</summary>
    public IReadOnlyList<bool>? CharacterTags { get; init; }

    /// <summary>Twin batches built, and the sum of their twins' mean Jaccard overlap with the positive (for the dashboard).</summary>
    public long TwinBatches, TwinMisses;
    public double TwinJaccardSum;

    private long _batches;

    private bool TwinCandidate(IReadOnlyList<int> tags) =>
        tags.Count is >= 8 and <= 80 && tags.Count(t => CharacterTags![t]) <= 1;

    /// <summary>One positive of an anchor from <see cref="TwinTags"/> and up to <paramref name="twins"/> images without it, most similar first; null if none found.</summary>
    public SampledImage[]? SampleTwins(int twins, int attempts = 16)
    {
        for (var attempt = 0; attempt < attempts; attempt++)
        {
            var anchor = TwinTags![_random.Next(TwinTags.Count)];
            var positives = _statistics.Positives[anchor];
            var positive = positives[_random.Next(positives.Length)];
            var tags = _statistics.ImageTagRows[positive];
            if (!TwinCandidate(tags))
                continue;
            var matched = FindTwins(tags, anchor, twins, positive);
            if (matched.Length < Math.Min(3, twins))
                continue;
            TwinBatches++;
            TwinJaccardSum += matched.Average(m => m.Jaccard);
            return [new SampledImage(positive, anchor, IsHardNegative: false, IsTwin: true),
                .. matched.Select(m => new SampledImage(m.Image, anchor, IsHardNegative: false, IsTwin: true))];
        }
        TwinMisses++;
        return null;
    }

    /// <summary>
    /// Up to <paramref name="twins"/> images without <paramref name="anchor"/> whose tags overlap <paramref name="tags"/> most
    /// (Jaccard), at most one character each, from the images carrying its 8 rarest other tags; most similar first.
    /// </summary>
    public (int Image, double Jaccard)[] FindTwins(IReadOnlyList<int> tags, int anchor, int twins, int exclude = -1)
    {
        var own = tags.ToHashSet();
        var candidates = new HashSet<int>();
        foreach (var tag in tags.Where(t => t != anchor && (uint)t < (uint)_statistics.VocabularySize && _statistics.Positives[t].Length > 1)
                     .OrderBy(t => _statistics.Positives[t].Length).Take(8))
        {
            var images = _statistics.Positives[tag];
            for (var i = 0; i < Math.Min(500, images.Length); i++)
                candidates.Add(images.Length <= 500 ? images[i] : images[_random.Next(images.Length)]);
        }
        return candidates.Where(c => c != exclude)
            .Select(c => (Image: c, Tags: _statistics.ImageTagRows[c]))
            .Where(c => !c.Tags.Contains(anchor) && TwinCandidate(c.Tags))
            .Select(c =>
            {
                var shared = c.Tags.Count(own.Contains);
                return (c.Image, Jaccard: shared / (double)(own.Count + c.Tags.Count - shared));
            })
            .OrderByDescending(c => c.Jaccard).Take(twins).ToArray();
    }

    public SampledImage[] SampleBatch(int batchSize)
    {
        if (TwinEvery > 0 && _batches++ % TwinEvery == 0 && SampleTwins(batchSize - 1) is { } twinBatch)
        {
            if (twinBatch.Length == batchSize)
                return twinBatch;
            return [.. twinBatch, .. SampleRegular(batchSize - twinBatch.Length)];
        }
        return SampleRegular(batchSize);
    }

    private SampledImage[] SampleRegular(int batchSize)
    {

        var batch = new List<SampledImage>(batchSize);
        var focus = _focus;
        if (ContextPairsPerBatch > 0)
        {
            _contextPairTags ??= (ContextPairTags ?? Enumerable.Range(0, _statistics.VocabularySize).ToArray())
                .Where(t => _statistics.ContextSources[t].Length > 0).ToArray();
            for (var i = 0; i < ContextPairsPerBatch && _contextPairTags.Length > 0 && batch.Count + 2 <= batchSize; i++)
            {
                var anchor = _contextPairTags[_random.Next(_contextPairTags.Length)];
                if (_statistics.SampleContextPair(anchor, _random) is var (with, without))
                {
                    var pair = batch.Count;
                    batch.Add(new SampledImage(with, anchor, IsHardNegative: false, ContextPair: pair));
                    batch.Add(new SampledImage(without, anchor, IsHardNegative: false, IsContextNegative: true, ContextPair: pair));
                }
            }
        }
        if (ExclusivePerBatch > 0)
        {
            _contextPairTags ??= (ContextPairTags ?? Enumerable.Range(0, _statistics.VocabularySize).ToArray())
                .Where(t => _statistics.ContextSources[t].Length > 0).ToArray();
            for (var i = 0; i < ExclusivePerBatch && _contextPairTags.Length > 0 && batch.Count < batchSize; i++)
            {
                var anchor = _contextPairTags[_random.Next(_contextPairTags.Length)];
                if (i % 2 == 0 && _statistics.SampleContextNegative(anchor, _random) is int without)
                    batch.Add(new SampledImage(without, anchor, IsHardNegative: false, IsContextNegative: true));
                else if (i % 2 == 1 && ExclusivePositive(anchor) is int with)
                    batch.Add(new SampledImage(with, anchor, IsHardNegative: false));
            }
        }
        while (batch.Count < batchSize)
        {
            if (ExcluderShare > 0 && _random.NextDouble() < ExcluderShare)
            {
                // The ruled-out parents only (shoes, not each of its 50 kinds), so every rule gets the same share: drawn over
                // every ruled-out tag, half the anchors were kinds of shoe, and footwear AP fell in the A/B.
                _excludedTags ??= Enumerable.Range(0, _statistics.VocabularySize)
                    .Where(t => _statistics.Positives[t].Length > 0 && _statistics.Families.Excluders(t).Count > 0
                        && !_statistics.Families.KindOf(t).Any(a => _statistics.Families.Excluders(a).Count > 0)).ToArray();
                if (_excludedTags.Length > 0)
                {
                    var excluded = _excludedTags[_random.Next(_excludedTags.Length)];
                    var excludedPositives = _statistics.Positives[excluded];
                    for (var i = 0; i < _positivesPerAnchor && batch.Count < batchSize; i++)
                        batch.Add(SamplePositive(excluded, excludedPositives));
                    for (var i = 0; i < 2 && batch.Count < batchSize; i++)
                        if (SampleExcluderNegative(excluded) is int image)
                            batch.Add(new SampledImage(image, excluded, IsHardNegative: true));
                    continue;
                }
            }

            var focused = focus.Length > 0 && _random.NextDouble() < FocusShare;
            var anchor = focused ? focus[_random.Next(focus.Length)] : _anchorTags[_random.Next(_anchorTags.Length)];
            var positives = _statistics.Positives[anchor];

            for (var i = 0; i < _positivesPerAnchor && batch.Count < batchSize; i++)
                batch.Add(SamplePositive(anchor, positives));

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
