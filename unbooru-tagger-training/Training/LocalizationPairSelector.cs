using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>The (image, tag) pairs the per-location loss trains on, padded to the same count per image. All tensors are [batch, pairsPerImage].</summary>
/// <param name="TagRows">Vocabulary row of each pair (0 for padding).</param>
/// <param name="Labels">+1 / -1.</param>
/// <param name="Weights">Pair weight (0 for padding).</param>
/// <param name="FocalMask">1 = unverified negative (asymmetric-loss damped), 0 = full-strength.</param>
public sealed record LocalizationPairs(Tensor TagRows, Tensor Labels, Tensor Weights, Tensor FocalMask, float PositiveWeightSum) : IDisposable
{
    public LocalizationPairs To(Device device) => new(TagRows.Upload(device), Labels.Upload(device), Weights.Upload(device), FocalMask.Upload(device), PositiveWeightSum);

    public void Dispose()
    {
        TagRows.Dispose();
        Labels.Dispose();
        Weights.Dispose();
        FocalMask.Dispose();
    }
}

/// <summary>
/// Picks which (image, tag) pairs the per-location localization loss is worth computing on.
/// Scoring every tag at every location (the old dense loss) spent ~2/3 of each training step
/// on pairs where "where is it?" has no answer: tags that aren't on the image and that the
/// model already confidently rejects, and tags with no region at all (artists, series,
/// ratings, "solo", "upper_body"). Per image this keeps only:
/// <list type="bullet">
/// <item>every localizable positive — where is it?</item>
/// <item>every localizable certain negative (<see cref="PairWeightBuilder"/>: the anchor's
/// hard negative, mutually exclusive tags) — it must fire nowhere;</item>
/// <item>the <c>minedNegatives</c> highest-scoring localizable negatives by whole-image score —
/// the false positives the model is currently closest to making, where a wrongly lit-up
/// region is most likely.</item>
/// </list>
/// Likely-missing and untrainable pairs (weight 0) are never picked.
/// </summary>
public static class LocalizationPairSelector
{
    /// <summary>
    /// Pair widths are rounded up to a multiple of this (the padding at weight 0, as short rows already are). Unrounded, the width is
    /// the batch's widest image and changes nearly every step; on the B70 each new size makes the caching allocator reserve new blocks
    /// it never returns, so VRAM ratchets up until a wide step fails (run 4, 2026-11-03: 12 -> 24 GB in 10 minutes, then
    /// UR_RESULT_ERROR_OUT_OF_RESOURCES / xe "VM worker error: -12" after 40 minutes; run 5's locality term, 2026-11-01, the same).
    /// </summary>
    public const int WidthMultiple = 64;

    /// <summary>The widest unrounded pair row seen in this process (for the dashboard).</summary>
    public static int WidestSeen;

    /// <summary>
    /// An image's labelled pairs (positives and certain negatives) are capped at this many: every positive is kept, and the
    /// certain negatives are sampled down to the room left, their weights scaled by (all / kept) so the image's expected
    /// negative loss is unchanged. Padding alone didn't bound the peak: some images carry hundreds of certain negatives
    /// (context negatives of a character with many context targets, an excludes rule over a whole family), the widest row
    /// seen was 788, and each wider bucket reserved more VRAM (an A/B arm reached 29.4 of 32 GB, 2026-11-03).
    /// </summary>
    public const int MaxLabelledPairs = 192;

    /// <summary>Images whose labelled pairs were sampled down by <see cref="CapLabelled"/> in this process.</summary>
    public static long CappedImages;

    /// <summary>
    /// <paramref name="pairs"/> capped at <paramref name="cap"/> (see <see cref="MaxLabelledPairs"/>): all positives
    /// (label &gt; 0 at <paramref name="offset"/>), then a random sample of the rest. Returns the kept pairs and the weight
    /// scale for the kept non-positives (1 when nothing was dropped).
    /// </summary>
    public static (List<int> Kept, float NegativeScale) CapLabelled(List<int> pairs, float[] labels, int offset, int cap = MaxLabelledPairs)
    {
        if (pairs.Count <= cap)
            return (pairs, 1f);
        var positives = pairs.Where(t => labels[offset + t] > 0).ToList();
        var negatives = pairs.Where(t => labels[offset + t] <= 0).ToList();
        var room = Math.Max(0, cap - positives.Count);
        if (negatives.Count <= room)
            return (pairs, 1f);
        Interlocked.Increment(ref CappedImages);
        var kept = negatives.OrderBy(_ => Random.Shared.Next()).Take(room).ToList();
        return ([.. positives, .. kept], room > 0 ? (float)negatives.Count / room : 0f);
    }

    /// <summary><paramref name="width"/> rounded up to <see cref="WidthMultiple"/>, recording the widest seen.</summary>
    public static int Padded(int width)
    {
        if (width > WidestSeen)
            WidestSeen = width;
        return (Math.Max(1, width) + WidthMultiple - 1) / WidthMultiple * WidthMultiple;
    }

    /// <param name="targets">This batch's per-pair targets (CPU).</param>
    /// <param name="wholeImageLogits">[batch × vocabulary] whole-image logits, row-major — only used to rank negatives.</param>
    /// <param name="localizable">Per vocabulary row: whether the tag has a region to point at.</param>
    public static LocalizationPairs Select(PairTargets targets, float[] wholeImageLogits, bool[] localizable, int minedNegatives)
    {
        var batch = (int)targets.Labels.shape[0];
        var vocabularySize = (int)targets.Labels.shape[1];
        var labels = targets.Labels.data<float>().ToArray();
        var weights = targets.Weights.data<float>().ToArray();
        var focal = targets.FocalMask.data<float>().ToArray();

        var perImage = new List<int>[batch];
        var negativeScale = new float[batch];
        var labelledCount = new int[batch];
        for (var i = 0; i < batch; i++)
        {
            var offset = i * vocabularySize;
            var chosen = new List<int>();
            var candidates = new List<int>();
            for (var t = 0; t < vocabularySize; t++)
            {
                var index = offset + t;
                if (!localizable[t] || weights[index] <= 0)
                    continue;
                if (labels[index] > 0 || focal[index] == 0)
                    chosen.Add(t); // positive, or a certain negative
                else
                    candidates.Add(t);
            }
            (chosen, negativeScale[i]) = CapLabelled(chosen, labels, offset);
            labelledCount[i] = chosen.Count;

            if (minedNegatives > 0 && candidates.Count > 0)
                chosen.AddRange(candidates
                    .OrderByDescending(t => wholeImageLogits[offset + t])
                    .Take(minedNegatives));
            perImage[i] = chosen;
        }

        var width = Padded(perImage.Max(p => p.Count));
        var tagRows = new long[batch * width];
        var pairLabels = new float[batch * width];
        var pairWeights = new float[batch * width];
        var pairFocal = new float[batch * width];
        Array.Fill(pairLabels, -1f);
        var positiveWeightSum = 0f;
        for (var i = 0; i < batch; i++)
            for (var k = 0; k < perImage[i].Count; k++)
            {
                var t = perImage[i][k];
                var source = i * vocabularySize + t;
                var target = i * width + k;
                tagRows[target] = t;
                pairLabels[target] = labels[source];
                // Certain negatives sampled down by CapLabelled count for the ones dropped (mined negatives, after them, don't).
                pairWeights[target] = weights[source] * (k < labelledCount[i] && labels[source] <= 0 ? negativeScale[i] : 1f);
                pairFocal[target] = focal[source];
                if (labels[source] > 0)
                    positiveWeightSum += weights[source];
            }

        long[] shape = [batch, width];
        return new LocalizationPairs(tensor(tagRows, shape), tensor(pairLabels, shape), tensor(pairWeights, shape), tensor(pairFocal, shape), positiveWeightSum);
    }

    /// <summary>
    /// <see cref="Select"/>'s pairs, on the device and without reading <paramref name="scores"/> back, so a training step never
    /// waits for it: the positives and certain negatives come from the labels (<paramref name="cpuTargets"/>, on the CPU), and
    /// only the mined negatives need the scores, which a top-k picks on the device. Padding is laid out differently (the mined
    /// negatives take the last <paramref name="minedNegatives"/> columns, any short of candidates at weight 0); the pairs are the same.
    /// </summary>
    /// <param name="targets">The same targets on the device.</param>
    /// <param name="scores">[batch, vocabulary] whole-image scores on the device — only used to rank negatives.</param>
    /// <param name="labelled">Each image's pairs that came from the labels (positives and certain negatives), for <see cref="Mosaic.TilePairsOnDevice"/>.</param>
    public static LocalizationPairs SelectOnDevice(PairTargets cpuTargets, PairTargets targets, Tensor scores, bool[] localizable, int minedNegatives,
        out List<int>[] labelled)
    {
        var batch = (int)cpuTargets.Labels.shape[0];
        var vocabularySize = (int)cpuTargets.Labels.shape[1];
        var labels = cpuTargets.Labels.data<float>().ToArray();
        var weights = cpuTargets.Weights.data<float>().ToArray();
        var focal = cpuTargets.FocalMask.data<float>().ToArray();
        var device = scores.device;

        var perImage = new List<int>[batch];
        var negativeScale = new float[batch];
        var candidates = new bool[batch * vocabularySize];
        for (var i = 0; i < batch; i++)
        {
            var offset = i * vocabularySize;
            perImage[i] = [];
            for (var t = 0; t < vocabularySize; t++)
            {
                var index = offset + t;
                if (!localizable[t] || weights[index] <= 0)
                    continue;
                if (labels[index] > 0 || focal[index] == 0)
                    perImage[i].Add(t); // positive, or a certain negative
                else
                    candidates[index] = true;
            }
            (perImage[i], negativeScale[i]) = CapLabelled(perImage[i], labels, offset);
        }

        var width = Padded(perImage.Max(p => p.Count));
        var tagRows = new long[batch * width];
        var pairLabels = new float[batch * width];
        var pairWeights = new float[batch * width];
        var pairFocal = new float[batch * width];
        Array.Fill(pairLabels, -1f);
        var positiveWeightSum = 0f;
        for (var i = 0; i < batch; i++)
            for (var k = 0; k < perImage[i].Count; k++)
            {
                var t = perImage[i][k];
                var source = i * vocabularySize + t;
                var target = i * width + k;
                tagRows[target] = t;
                pairLabels[target] = labels[source];
                pairWeights[target] = weights[source] * (labels[source] <= 0 ? negativeScale[i] : 1f);
                pairFocal[target] = focal[source];
                if (labels[source] > 0)
                    positiveWeightSum += weights[source];
            }

        labelled = perImage;
        long[] shape = [batch, width];
        var rows = tensor(tagRows, shape).Upload(device);
        var pairLabelTensor = tensor(pairLabels, shape).Upload(device);
        var pairWeightTensor = tensor(pairWeights, shape).Upload(device);
        var pairFocalTensor = tensor(pairFocal, shape).Upload(device);
        if (minedNegatives > 0)
        {
            using var scope = NewDisposeScope();
            using var _ = no_grad();
            var candidateMask = tensor(candidates, [batch, vocabularySize]).Upload(device);
            // An explicit, scoped Scalar: run 4 died once (2026-11-04, ~2.07M steps) in masked_fill's Scalar::toFloat check, as if
            // the implicitly converted Scalar had been finalized while the native call still used it.
            using TorchSharp.Scalar negativeInfinity = float.NegativeInfinity;
            var (values, mined) = scores.detach().masked_fill(candidateMask.logical_not(), negativeInfinity).topk(Math.Min(minedNegatives, vocabularySize), 1);
            var real = values.isfinite().to_type(ScalarType.Float32);
            var combined = (
                cat([rows, mined], 1),
                cat([pairLabelTensor, full_like(values, -1f, dtype: ScalarType.Float32)], 1),
                cat([pairWeightTensor, targets.Weights.gather(1, mined) * real], 1),
                cat([pairFocalTensor, targets.FocalMask.gather(1, mined)], 1));
            rows.Dispose();
            pairLabelTensor.Dispose();
            pairWeightTensor.Dispose();
            pairFocalTensor.Dispose();
            (rows, pairLabelTensor, pairWeightTensor, pairFocalTensor) = combined;
            scope.MoveToOuter(rows, pairLabelTensor, pairWeightTensor, pairFocalTensor);
        }
        return new LocalizationPairs(rows, pairLabelTensor, pairWeightTensor, pairFocalTensor, positiveWeightSum);
    }
}
