using UnbooruTagger.Core.Dataset;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Per-tag structure of one image set (normally the training split), built once and
/// shared by <see cref="TagCentricBatchSampler"/>, <see cref="PairWeightBuilder"/> and
/// <see cref="TagRankingEvaluator"/>:
/// <list type="bullet">
/// <item><see cref="Positives"/> — every image carrying each tag (an inverted index).</item>
/// <item><see cref="ImpliedBy"/> — for a tag s, the tags t that appear on at least
/// <c>missingThreshold</c> of s's images (large_breasts → breasts, a character → her hair
/// color). An image with s but without t is more likely a missing tag than a real
/// negative, so it shouldn't be trained as a "no" at all.</item>
/// <item><see cref="ContextSources"/> — for a tag t, tags s that usually but not nearly
/// always come with it: <c>impliedThreshold</c> ≤ P(t|s) &lt; <c>missingThreshold</c> (a
/// character who usually wears a hat → hat). Images with s but without t are often real
/// absences (the hat is off) and are the most informative negatives the corpus has: the
/// context is there, the object isn't. Without them a model can learn "this character,
/// therefore hat" — run 2's epoch-1 classifier said "hat" on 89% of gold crops that held
/// no hat. Some are still missing labels, so they train as unverified (focal-damped)
/// negatives, never certain ones.</item>
/// <item><see cref="HardNegativeSources"/> — for a tag t, tags s that show up alongside t
/// often (P(s|t) ≥ <c>hardNegativeMinRatio</c>) AND have plenty of images without t. Images
/// with s but not t are the visually-confusable negatives the crawler deliberately
/// collected (same definition as the crawler's <c>TagCooccurrenceIndex.FindHardNegativeSources</c>).</item>
/// </list>
/// Co-occurrence is estimated from a bounded random sample of each tag's own images
/// (<c>samplesPerTag</c>), not every image pair in the corpus: an exact pair count over
/// millions of images × ~40 tags each is billions of increments and a pair table too big
/// to hold, while a few hundred samples per tag already pin a conditional frequency to a
/// few percent — plenty to decide "≥ 0.6" or "≥ 0.3".
/// </summary>
public sealed class TagStatistics
{
    public int VocabularySize { get; }

    /// <summary>[tag] → indices (into the image list this was built from) of every image carrying that tag.</summary>
    public int[][] Positives { get; }

    /// <summary>[tag s] → tags t (≠ s) present on at least the missing-threshold fraction of s's images.</summary>
    public int[][] ImpliedBy { get; }

    /// <summary>[tag t] → tags s with impliedThreshold ≤ P(t|s) &lt; missingThreshold and plenty of images without t, most counter-examples first.</summary>
    public int[][] ContextSources { get; }

    /// <summary>[tag t] → tags s whose images-without-t make good hard negatives for t, strongest association first.</summary>
    public int[][] HardNegativeSources { get; }

    public IReadOnlyList<IReadOnlyList<int>> ImageTagRows { get; }

    /// <summary>Optional: which sites each image came from, and which sites use each tag (TagAgreement.LoadSiteCoverage).</summary>
    public IReadOnlyList<BooruSites>? ImageSites { get; private init; }
    public IReadOnlyList<BooruSites>? TagSites { get; private init; }

    /// <summary>
    /// True if no site <paramref name="image"/> came from uses <paramref name="tag"/>: it can
    /// never carry the tag, so its absence is no evidence either way.
    /// </summary>
    public bool IsSiteBlind(int image, int tag) =>
        ImageSites is not null && TagSites is not null && (ImageSites[image] & TagSites[tag]) == BooruSites.None;

    private TagStatistics(int vocabularySize, int[][] positives, int[][] impliedBy, int[][] contextSources, int[][] hardNegativeSources, IReadOnlyList<IReadOnlyList<int>> imageTagRows)
    {
        VocabularySize = vocabularySize;
        Positives = positives;
        ImpliedBy = impliedBy;
        ContextSources = contextSources;
        HardNegativeSources = hardNegativeSources;
        ImageTagRows = imageTagRows;
    }

    public bool HasTag(int image, int tag)
    {
        var rows = ImageTagRows[image];
        for (var i = 0; i < rows.Count; i++)
            if (rows[i] == tag)
                return true;
        return false;
    }

    public static TagStatistics Build(
        IReadOnlyList<IReadOnlyList<int>> imageTagRows,
        int vocabularySize,
        int samplesPerTag = 256,
        double impliedThreshold = 0.6,
        double hardNegativeMinRatio = 0.3,
        int hardNegativeMinCounterExamples = 15,
        int maxHardNegativeSources = 8,
        int seed = 0,
        double? missingThreshold = null,
        int maxContextSources = 8,
        IReadOnlyList<double>? tagRecall = null,
        double missingPosterior = 0.5,
        IReadOnlyList<BooruSites>? imageSites = null,
        IReadOnlyList<BooruSites>? tagSites = null)
    {
        // Without a separate missing threshold every implied pair is "likely missing" (the
        // original behavior) and there is no context band.
        var missing = missingThreshold ?? impliedThreshold;

        // Is an image with s but not t (P(t|s) = p) more likely a missing label than a real
        // absence? With a per-tag tagging recall r (TagAgreement: how often taggers list t
        // when it's there), P(t present | not tagged) = p(1-r) / (p(1-r) + 1-p); likely
        // missing when that reaches missingPosterior. Without recall, the fixed cutoff.
        bool IsMissing(int t, double p)
        {
            if (p < impliedThreshold)
                return false;
            if (tagRecall is null || t >= tagRecall.Count || double.IsNaN(tagRecall[t]))
                return p >= missing;
            var hidden = p * (1 - tagRecall[t]);
            return hidden / (hidden + 1 - p) >= missingPosterior;
        }
        var counts = new int[vocabularySize];
        foreach (var rows in imageTagRows)
            foreach (var row in rows)
                if ((uint)row < (uint)vocabularySize)
                    counts[row]++;

        var positives = new int[vocabularySize][];
        for (var t = 0; t < vocabularySize; t++)
            positives[t] = new int[counts[t]];

        var fill = new int[vocabularySize];
        for (var image = 0; image < imageTagRows.Count; image++)
            foreach (var row in imageTagRows[image])
                if ((uint)row < (uint)vocabularySize)
                    positives[row][fill[row]++] = image;

        // conditional[s] = (t, P(t|s)) for every t seen on at least
        // min(impliedThreshold, hardNegativeMinRatio) of s's sampled images — the only
        // ratios either consumer below ever asks about.
        var keepRatio = Math.Min(impliedThreshold, hardNegativeMinRatio);
        var conditional = new (int Tag, float Ratio)[vocabularySize][];
        Parallel.For(0, vocabularySize, s =>
        {
            var images = positives[s];
            if (images.Length == 0)
            {
                conditional[s] = [];
                return;
            }

            var random = new Random(HashCode.Combine(seed, s));
            var sampleCount = Math.Min(samplesPerTag, images.Length);
            var tally = new Dictionary<int, int>();
            for (var i = 0; i < sampleCount; i++)
            {
                var image = sampleCount == images.Length ? images[i] : images[random.Next(images.Length)];
                foreach (var t in imageTagRows[image])
                    if (t != s)
                        tally[t] = tally.GetValueOrDefault(t) + 1;
            }

            conditional[s] = tally
                .Select(kv => (Tag: kv.Key, Ratio: kv.Value / (float)sampleCount))
                .Where(x => x.Ratio >= keepRatio)
                .ToArray();
        });

        var impliedBy = new int[vocabularySize][];
        var hardNegativeSources = new int[vocabularySize][];
        Parallel.For(0, vocabularySize, t =>
        {
            impliedBy[t] = conditional[t].Where(x => IsMissing(x.Tag, x.Ratio)).Select(x => x.Tag).Order().ToArray();

            // conditional[t] holds P(s|t) for each s: "s shows up alongside t this often".
            // Counter-examples (images with s but without t) ≈ count(s) - P(s|t)·count(t).
            hardNegativeSources[t] = conditional[t]
                .Where(x => x.Ratio >= hardNegativeMinRatio)
                .Select(x => (x.Tag, x.Ratio, CounterExamples: counts[x.Tag] - x.Ratio * counts[t]))
                .Where(x => x.CounterExamples >= hardNegativeMinCounterExamples)
                .OrderByDescending(x => x.Ratio)
                .ThenByDescending(x => x.CounterExamples)
                .ThenBy(x => x.Tag)
                .Take(maxHardNegativeSources)
                .Select(x => x.Tag)
                .ToArray();
        });

        // conditional[s] holds P(t|s); invert the context band into t → sources s.
        var contextCandidates = new List<(int Source, float CounterExamples)>[vocabularySize];
        for (var t = 0; t < vocabularySize; t++)
            contextCandidates[t] = [];
        for (var s = 0; s < vocabularySize; s++)
            foreach (var (t, ratio) in conditional[s])
                if (ratio >= impliedThreshold && !IsMissing(t, ratio))
                    contextCandidates[t].Add((s, counts[s] * (1 - ratio)));
        var contextSources = contextCandidates
            .Select(list => list
                .Where(x => x.CounterExamples >= hardNegativeMinCounterExamples)
                .OrderByDescending(x => x.CounterExamples)
                .ThenBy(x => x.Source)
                .Take(maxContextSources)
                .Select(x => x.Source)
                .ToArray())
            .ToArray();

        return new TagStatistics(vocabularySize, positives, impliedBy, contextSources, hardNegativeSources, imageTagRows)
        {
            ImageSites = imageSites,
            TagSites = tagSites,
        };
    }

    /// <summary>
    /// One random hard negative for <paramref name="tag"/>: an image carrying one of its
    /// <see cref="HardNegativeSources"/> but not <paramref name="tag"/> itself, nor any tag
    /// that usually implies it (that would be a likely-missing label, not a real negative), nor
    /// a context negative (those train only as unverified negatives, never certain ones).
    /// Falls back to any random image without the tag when the tag has no qualifying
    /// sources or every attempt hits a positive. Null only if every attempt fails.
    /// </summary>
    public int? SampleHardNegative(int tag, Random random, int attempts = 16)
    {
        var sources = HardNegativeSources[tag];
        for (var attempt = 0; attempt < attempts; attempt++)
        {
            int image;
            if (sources.Length > 0 && attempt < attempts / 2)
            {
                var sourceImages = Positives[sources[random.Next(sources.Length)]];
                if (sourceImages.Length == 0)
                    continue;
                image = sourceImages[random.Next(sourceImages.Length)];
            }
            else
            {
                image = random.Next(ImageTagRows.Count);
            }

            if (!HasTag(image, tag) && !IsLikelyMissing(image, tag) && !IsContextNegative(image, tag) && !IsSiteBlind(image, tag))
                return image;
        }

        return null;
    }

    /// <summary>True if <paramref name="image"/> carries one of <paramref name="tag"/>'s <see cref="ContextSources"/>.</summary>
    public bool IsContextNegative(int image, int tag)
    {
        var sources = ContextSources[tag];
        if (sources.Length == 0)
            return false;
        foreach (var s in ImageTagRows[image])
            if (Array.IndexOf(sources, s) >= 0)
                return true;
        return false;
    }

    /// <summary>
    /// One random context negative for <paramref name="tag"/>: an image carrying one of its
    /// <see cref="ContextSources"/> but not the tag itself and not a likely-missing label.
    /// Null if the tag has no context sources or every attempt fails.
    /// </summary>
    // More attempts than hard negatives: a source can carry the tag up to 90% of the time.
    public int? SampleContextNegative(int tag, Random random, int attempts = 64)
    {
        var sources = ContextSources[tag];
        if (sources.Length == 0)
            return null;
        for (var attempt = 0; attempt < attempts; attempt++)
        {
            var sourceImages = Positives[sources[random.Next(sources.Length)]];
            if (sourceImages.Length == 0)
                continue;
            var image = sourceImages[random.Next(sourceImages.Length)];
            if (!HasTag(image, tag) && !IsLikelyMissing(image, tag) && !IsSiteBlind(image, tag))
                return image;
        }

        return null;
    }

    /// <summary>True if <paramref name="image"/> lacks <paramref name="tag"/> but carries some tag that nearly always implies it.</summary>
    public bool IsLikelyMissing(int image, int tag)
    {
        foreach (var s in ImageTagRows[image])
        {
            if ((uint)s >= (uint)VocabularySize)
                continue;
            if (Array.BinarySearch(ImpliedBy[s], tag) >= 0)
                return true;
        }

        return false;
    }
}
