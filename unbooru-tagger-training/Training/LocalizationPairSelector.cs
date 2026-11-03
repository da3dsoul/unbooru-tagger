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

            if (minedNegatives > 0 && candidates.Count > 0)
                chosen.AddRange(candidates
                    .OrderByDescending(t => wholeImageLogits[offset + t])
                    .Take(minedNegatives));
            perImage[i] = chosen;
        }

        var width = Math.Max(1, perImage.Max(p => p.Count));
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
                pairWeights[target] = weights[source];
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
        }

        var width = Math.Max(1, perImage.Max(p => p.Count));
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
                pairWeights[target] = weights[source];
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
            var (values, mined) = scores.detach().masked_fill(candidateMask.logical_not(), float.NegativeInfinity).topk(Math.Min(minedNegatives, vocabularySize), 1);
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
