using static TorchSharp.torch;

namespace UnbooruTagger.Training.Model;

/// <summary>
/// Whole-image tag scores pooled per tag instead of from one averaged embedding
/// (<see cref="ImageTower.ClassificationPoolingTemperature"/>). The averaged embedding makes a tag's score the
/// mean of its per-location scores, so an object covering 1-2% of the map barely moves it, however clearly it's
/// there; the cheaper fit is to raise every location from outfit and genre cues. Run 4's held-out scores showed
/// exactly that for gloves: a median of 0.59 on images with gloves and 0.47 without, misses concentrated on
/// nude/monochrome images, false positives on armor and bodysuits.
/// <para>
/// For the given rows (localizable tags), the full spatial map is averaged into a <see cref="Grid"/>²-cell grid
/// over the image's content and the score is a log-sum-exp over the cells at a temperature, in raw dot-product
/// units before the learned scale and bias, as <see cref="Training.SigmoidContrastiveLoss.ComputeLocalizedPairs"/>
/// pools. One clear cell can then carry the score. The mean over cells is subtracted, so a high temperature gives
/// back the average. Every other tag keeps the averaged embedding. Training pools only the rows
/// <see cref="TrainedRows"/> picks each step; evaluation pools every given row.
/// </para>
/// </summary>
public static class ClassificationPooling
{
    /// <summary>
    /// The temperature annealed from <paramref name="StartTemperature"/> to <paramref name="TargetTemperature"/> over
    /// <paramref name="AnnealSteps"/> global steps from <paramref name="StartStep"/> (when pooling was turned on),
    /// log-linearly. Switching from the averaged embedding straight to a low temperature changed every localizable
    /// tag's scores at once and cost 6,000 steps of mAP (0.577 to 0.537, at 0.25 and at 1 alike). A high temperature
    /// is the average (raw scores sit around 1.1-1.2, so 8 is well above their spread), so annealing starts where the
    /// run already is. Saved in the model config, so a resume continues it.
    /// </summary>
    public sealed record Schedule(long StartStep, float StartTemperature, float TargetTemperature, int AnnealSteps)
    {
        public float At(long globalStep)
        {
            if (AnnealSteps <= 0)
                return TargetTemperature;
            var progress = Math.Clamp((globalStep - StartStep) / (double)AnnealSteps, 0, 1);
            return (float)Math.Exp(Math.Log(StartTemperature) + progress * (Math.Log(TargetTemperature) - Math.Log(StartTemperature)));
        }
    }

    /// <summary>Cells per side: 32px at 512px input.</summary>
    public const int Grid = 16;

    /// <summary>[B, D, Grid²] content-weighted cell averages of <paramref name="spatial"/>, and [B, Grid²] which cells hold any content.</summary>
    /// <param name="spatialMask">[B, 1, H, W], 1 over the image's content (<see cref="SpatialMask"/>).</param>
    public static (Tensor Cells, Tensor Valid) Cells(Tensor spatial, Tensor spatialMask)
    {
        var weighted = nn.functional.adaptive_avg_pool2d(spatial * spatialMask, [Grid, Grid]);
        var coverage = nn.functional.adaptive_avg_pool2d(spatialMask, [Grid, Grid]);
        return ((weighted / coverage.clamp_min(1e-6f)).flatten(2), coverage.flatten(1).gt(0));
    }

    /// <summary>
    /// The pooled rows to train with gradients this step: every pooled row that is a positive or a full-strength
    /// negative for some batch image, plus each image's <paramref name="minedPerImage"/> highest-scoring pooled
    /// negatives (scored without gradients). Pooling all ~16K localizable tags with gradients ran the B70 out of
    /// memory; the rest of the vocabulary trains on the averaged score, as before pooling existed, and a tag whose
    /// pooled score climbs is mined the next time its image comes up (the same mining as
    /// <see cref="Training.LocalizationPairSelector"/>).
    /// </summary>
    /// <param name="targets">The batch's targets, on the CPU.</param>
    public static Tensor TrainedRows(Tensor cells, Tensor valid, Tensor tagEmbeddings, long[] pooledRows, Training.PairTargets targets, float temperature, int minedPerImage)
    {
        var batch = (int)targets.Labels.shape[0];
        var vocabulary = (int)targets.Labels.shape[1];
        // [B, K] pooled scores, a chunk of rows at a time to keep the peak small; the constant shift
        // doesn't change the ranking, so it's left out.
        var scores = new float[batch * pooledRows.Length];
        using (no_grad())
        {
            const int chunk = 2048;
            for (var start = 0; start < pooledRows.Length; start += chunk)
            {
                using var scope = NewDisposeScope();
                var count = Math.Min(chunk, pooledRows.Length - start);
                var rows = tensor(pooledRows.AsSpan(start, count).ToArray(), device: cells.device);
                var dots = cells.transpose(1, 2).matmul(tagEmbeddings.index_select(0, rows).t()) / temperature;
                var part = dots.masked_fill(valid.logical_not().unsqueeze(2), float.NegativeInfinity).logsumexp(1).cpu().data<float>().ToArray();
                for (var i = 0; i < batch; i++)
                    Array.Copy(part, i * count, scores, i * pooledRows.Length + start, count);
            }
        }

        var labels = targets.Labels.data<float>().ToArray();
        var weights = targets.Weights.data<float>().ToArray();
        var focal = targets.FocalMask.data<float>().ToArray();
        var chosen = new HashSet<long>();
        var negatives = new List<(float Score, long Row)>();
        for (var i = 0; i < batch; i++)
        {
            negatives.Clear();
            for (var j = 0; j < pooledRows.Length; j++)
            {
                var at = i * vocabulary + (int)pooledRows[j];
                if (weights[at] <= 0)
                    continue;
                if (labels[at] > 0 || focal[at] == 0)
                    chosen.Add(pooledRows[j]);
                else
                    negatives.Add((scores[i * pooledRows.Length + j], pooledRows[j]));
            }
            foreach (var (_, row) in negatives.OrderByDescending(n => n.Score).Take(minedPerImage))
                chosen.Add(row);
        }

        return tensor(chosen.Order().ToArray(), device: cells.device);
    }

    /// <summary>
    /// [B, V] raw scores (dot-product units, before <see cref="LogitScaleBias"/>): <c>pooled · tag</c> for every
    /// tag, replaced for <paramref name="pooledRows"/> by <c>T · (logsumexp over valid cells of cell · tag / T − log n)</c>.
    /// </summary>
    public static Tensor RawScores(Tensor pooled, Tensor cells, Tensor valid, Tensor tagEmbeddings, Tensor pooledRows, float temperature)
    {
        var mean = pooled.matmul(tagEmbeddings.t());
        if (pooledRows.numel() == 0)
            return mean;
        var dots = cells.transpose(1, 2).matmul(tagEmbeddings.index_select(0, pooledRows).t()) / temperature; // [B, L, K]
        dots = dots.masked_fill(valid.logical_not().unsqueeze(2), float.NegativeInfinity);
        var count = valid.sum(1, keepdim: true).to_type(ScalarType.Float32).clamp_min(1f);
        var pooledScores = temperature * (dots.logsumexp(1) - count.log());
        return mean.index_copy(1, pooledRows, pooledScores);
    }
}
