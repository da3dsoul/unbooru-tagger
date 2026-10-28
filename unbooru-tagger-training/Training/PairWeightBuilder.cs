using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Vocabulary;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>Per-(image, tag) training targets for <see cref="SigmoidContrastiveLoss.ComputeWeighted"/>, all [batchImages, vocabularySize].</summary>
/// <param name="Labels">+1 where the image carries the tag, -1 otherwise.</param>
/// <param name="Weights">How much the pair counts: 0 drops it from the loss entirely.</param>
/// <param name="FocalMask">1 for an unverified negative (damped by the asymmetric focal term), 0 for a positive or a certain negative (plain full-strength BCE).</param>
/// <param name="PositiveWeightSum">Sum of <paramref name="Weights"/> over positive pairs — the loss normalizer.</param>
/// <param name="LogitOffsets">Optional: added to the whole-image logits in the loss only (<see cref="ContextPrior"/>).</param>
public sealed record PairTargets(Tensor Labels, Tensor Weights, Tensor FocalMask, float PositiveWeightSum, Tensor? LogitOffsets = null) : IDisposable
{
    public PairTargets To(Device device) => new(Labels.to(device), Weights.to(device), FocalMask.to(device), PositiveWeightSum, LogitOffsets?.to(device));

    public void Dispose()
    {
        Labels.Dispose();
        Weights.Dispose();
        FocalMask.Dispose();
        LogitOffsets?.Dispose();
    }
}

/// <summary>
/// Turns a batch's tag lists into per-pair targets that say not just positive/negative but
/// how sure the corpus actually is about each pair — the old <see cref="BatchLabelBuilder"/>
/// treated every untagged tag as an equally certain "no", which booru data can't support
/// (taggers routinely leave tags off) and which drowned each image's ~40 positives in
/// ~25K equally-weighted negatives.
///
/// Each pair falls into one category; later rules override earlier ones:
/// <list type="number">
/// <item><b>Unverified negative</b> (default): tag absent, nothing else known. Weight 1,
/// focal-damped — a confident "no" here costs nothing, an uncertain one only a little.</item>
/// <item><b>Anchor hard negative</b>: the image was sampled as a hard negative for this tag
/// (<see cref="TagCentricBatchSampler"/>). Full-strength, <c>certainNegativeWeight</c>.</item>
/// <item><b>Hard-negative source</b> (only with <c>hardNegativeSourceWeight</c> &gt; 0): the image
/// carries one of the tag's <see cref="TagStatistics.HardNegativeSources"/> (a tag that often comes
/// with it, whose images the crawler found often lack it) but not the tag. The same evidence that
/// makes a sampled anchor hard negative certain, applied to every pair in the batch instead of
/// only the anchor's: unverified negatives are focal-damped, so a wrong "yes" at p = 0.6 is pushed
/// down about 12x less than a missed "yes" is pushed up, and the tags near a plausible one
/// (gloves on an outfit that often has them) were barely corrected. Full-strength, at this weight;
/// only for <c>contextNegativeTags</c>, for the same reason as context negatives.</item>
/// <item><b>Likely missing</b>: the tag is usually implied by one the image does carry
/// (<see cref="TagStatistics.ImpliedBy"/>). Weight 0 — don't punish the model for
/// predicting a tag the tagger probably forgot.</item>
/// <item><b>Context negative</b> (only with <c>contextNegativeWeight</c> &gt; 0): the image
/// carries one of the tag's <see cref="TagStatistics.ContextSources"/> (a character who usually
/// wears a hat) but not the tag, and it isn't likely missing. Full-strength,
/// <c>contextNegativeWeight</c>: the focal damping let "this character, therefore hat" stand,
/// since a confident wrong "yes" there was nearly free. Off, these stay unverified negatives.
/// Only for <c>contextNegativeTags</c> (training passes the localizable ones): whole-image tags
/// like 1girl/2girls are often just untagged, and promoting their context negatives doubled
/// the held-out people-count contradictions (1girl + 2girls both called) in the A/B.</item>
/// <item><b>Exclusive</b>: the tag is mutually exclusive with one the image carries
/// (<see cref="ExclusiveTagGroups"/>). A verified absence: full-strength, <c>certainNegativeWeight</c>.</item>
/// <item><b>Exhaustively labeled</b>: <c>exhaustiveRows</c> (the cumulative rating labels)
/// are known for every image, so their absence is a verified "no", never "likely missing".
/// Without this, <c>rating:at_least_sensitive</c> (64% of images) was implied by common tags
/// like 1girl, its negatives were nearly all weight 0, and it learned to fire on everything.
/// Full-strength, <c>certainNegativeWeight</c>.</item>
/// <item><b>Site-blind</b>: no site the image came from uses the tag
/// (<see cref="TagStatistics.IsSiteBlind"/>: Gelbooru's silver_hair on a Danbooru-only image).
/// The image could never carry it, so the absence says nothing: weight 0.</item>
/// <item><b>Positive</b>: weight 1, full-strength.</item>
/// <item><b>Untrainable tag</b>: no training image carries it at all (stale vocabulary rows).
/// Weight 0 — there's nothing to learn, and its row would otherwise only ever be pushed
/// down.</item>
/// </list>
/// With a <c>contextPrior</c>, the targets also carry its per-pair logit offsets.
/// </summary>
public sealed class PairWeightBuilder(
    TagStatistics statistics,
    ExclusiveTagGroups exclusiveGroups,
    float certainNegativeWeight = 1f,
    IReadOnlyList<int>? exhaustiveRows = null,
    float contextNegativeWeight = 0f,
    IReadOnlyList<bool>? contextNegativeTags = null,
    ContextPrior? contextPrior = null,
    float hardNegativeSourceWeight = 0f)
{
    public int VocabularySize => statistics.VocabularySize;

    /// <summary>Tags not used by every site — the only ones the site-blind rule can apply to.</summary>
    private readonly int[] _siteRestricted = statistics.TagSites is { } tagSites
        ? Enumerable.Range(0, tagSites.Count).Where(t => tagSites[t] != (BooruSites.Danbooru | BooruSites.Gelbooru)).ToArray()
        : [];

    /// <summary>For a source tag s, the tags it is a context source for (the inverse of <see cref="TagStatistics.ContextSources"/>).</summary>
    private readonly int[][] _contextTargets = contextNegativeWeight > 0 ? InvertContextSources(statistics, contextNegativeTags) : [];

    /// <summary>For a tag s, the tags it is a hard-negative source for (the inverse of <see cref="TagStatistics.HardNegativeSources"/>).</summary>
    private readonly int[][] _hardNegativeTargets = hardNegativeSourceWeight > 0 ? Invert(statistics.HardNegativeSources, statistics.VocabularySize, contextNegativeTags) : [];

    private static int[][] Invert(int[][] sources, int vocabularySize, IReadOnlyList<bool>? eligible)
    {
        var targets = Enumerable.Range(0, vocabularySize).Select(_ => new List<int>()).ToArray();
        for (var t = 0; t < sources.Length; t++)
            if (eligible is null || (t < eligible.Count && eligible[t]))
                foreach (var s in sources[t])
                targets[s].Add(t);
        return targets.Select(l => l.ToArray()).ToArray();
    }

    private static int[][] InvertContextSources(TagStatistics statistics, IReadOnlyList<bool>? eligible)
    {
        var targets = Enumerable.Range(0, statistics.VocabularySize).Select(_ => new List<int>()).ToArray();
        for (var t = 0; t < statistics.ContextSources.Length; t++)
            if (eligible is null || (t < eligible.Count && eligible[t]))
                foreach (var s in statistics.ContextSources[t])
                targets[s].Add(t);
        return targets.Select(l => l.ToArray()).ToArray();
    }

    /// <param name="imageTagRows">Each batch image's tags.</param>
    /// <param name="sampled">Optional per-image sampling info, same order as <paramref name="imageTagRows"/> — only used to mark anchor hard negatives.</param>
    /// <param name="batchSites">Optional per-image sites, same order as <paramref name="imageTagRows"/> — enables the site-blind rule.</param>
    public PairTargets Build(IReadOnlyList<IReadOnlyList<int>> imageTagRows, IReadOnlyList<SampledImage>? sampled = null, IReadOnlyList<BooruSites>? batchSites = null)
    {
        var vocabularySize = statistics.VocabularySize;
        var count = imageTagRows.Count * vocabularySize;
        var labels = new float[count];
        var weights = new float[count];
        var focal = new float[count];
        Array.Fill(labels, -1f);
        Array.Fill(weights, 1f);
        Array.Fill(focal, 1f);

        var positiveWeightSum = 0f;
        for (var i = 0; i < imageTagRows.Count; i++)
        {
            var offset = i * vocabularySize;
            var tags = imageTagRows[i];

            if (sampled is not null && sampled[i].IsHardNegative)
                SetCertainNegative(offset + sampled[i].AnchorTag);

            if (_hardNegativeTargets.Length > 0)
                foreach (var s in tags)
                    if ((uint)s < (uint)vocabularySize)
                        foreach (var t in _hardNegativeTargets[s])
                        {
                            weights[offset + t] = Math.Max(weights[offset + t], hardNegativeSourceWeight);
                            focal[offset + t] = 0f;
                        }

            foreach (var s in tags)
            {
                if ((uint)s >= (uint)vocabularySize)
                    continue;
                foreach (var t in statistics.ImpliedBy[s])
                    weights[offset + t] = 0f;
            }

            if (_contextTargets.Length > 0)
                foreach (var s in tags)
                {
                    if ((uint)s >= (uint)vocabularySize)
                        continue;
                    foreach (var t in _contextTargets[s])
                        if (weights[offset + t] > 0)
                        {
                            weights[offset + t] = contextNegativeWeight;
                            focal[offset + t] = 0f;
                        }
                }

            foreach (var s in tags)
            {
                if ((uint)s >= (uint)vocabularySize)
                    continue;
                foreach (var t in exclusiveGroups.ExcludedBy(s))
                    SetCertainNegative(offset + t);
            }

            if (exhaustiveRows is not null)
                foreach (var r in exhaustiveRows)
                    if ((uint)r < (uint)vocabularySize)
                        SetCertainNegative(offset + r);

            if (batchSites is not null && _siteRestricted.Length > 0)
                foreach (var t in _siteRestricted)
                    if ((batchSites[i] & statistics.TagSites![t]) == BooruSites.None)
                        weights[offset + t] = 0f;

            foreach (var s in tags)
            {
                if ((uint)s >= (uint)vocabularySize)
                    continue;
                labels[offset + s] = 1f;
                weights[offset + s] = 1f;
                focal[offset + s] = 0f;
            }

            for (var t = 0; t < vocabularySize; t++)
            {
                if (statistics.Positives[t].Length == 0)
                    weights[offset + t] = 0f;
                else if (labels[offset + t] > 0)
                    positiveWeightSum += weights[offset + t];
            }
        }

        var shape = new long[] { imageTagRows.Count, vocabularySize };
        Tensor? offsets = null;
        if (contextPrior is not null)
        {
            var values = new float[count];
            for (var i = 0; i < imageTagRows.Count; i++)
                contextPrior.Fill(imageTagRows[i], values.AsSpan(i * vocabularySize, vocabularySize));
            offsets = tensor(values, shape);
        }
        return new PairTargets(tensor(labels, shape), tensor(weights, shape), tensor(focal, shape), positiveWeightSum, offsets);

        void SetCertainNegative(int index)
        {
            weights[index] = certainNegativeWeight;
            focal[index] = 0f;
        }
    }
}
