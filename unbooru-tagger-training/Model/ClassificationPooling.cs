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

    /// <summary>
    /// [B, L] pooled raw scores from [B, L, K] per-window (or per-cell) dots and [B, K] which windows are real:
    /// <c>T · (logsumexp(dot / T) − log n)</c>, or with <paramref name="topK"/> above 0 the mean of the k highest dots (an image
    /// with fewer real windows averages the ones it has), plus <paramref name="bottomWeight"/> times the mean of the k lowest
    /// (WILDCAT's negative evidence). -infinity for an image with none.
    /// </summary>
    /// <param name="valid">[B, K] which windows are real, or [B, L, K] which windows each tag pools.</param>
    public static Tensor Pool(Tensor dots, Tensor valid, float temperature, int topK, float bottomWeight = 0)
    {
        // [B, K] (every tag pools the same windows) or [B, L, K] (per tag).
        if (valid.dim() == 2)
            valid = valid.unsqueeze(1);
        var masked = dots.masked_fill(valid.logical_not(), float.NegativeInfinity);
        if (topK > 0)
        {
            var k = (int)Math.Min(topK, masked.shape[2]);
            Tensor Mean(Tensor values)
            {
                var real = values.isfinite();
                return where(real, values, zeros_like(values)).sum(2) / real.sum(2).to_type(ScalarType.Float32).clamp_min(1f);
            }
            var count = valid.sum(2);
            var pooled = Mean(masked.topk(k, dim: 2).values);
            if (bottomWeight > 0)
                pooled = pooled + bottomWeight * Mean(-(-masked.masked_fill(valid.logical_not(), float.PositiveInfinity)).topk(k, dim: 2).values);
            return where(count.gt(0), pooled, full_like(pooled, float.NegativeInfinity));
        }
        var n = valid.sum(2).to_type(ScalarType.Float32);
        return temperature * ((masked / temperature).logsumexp(2) - n.clamp_min(1f).log());
    }

    /// <summary><see cref="Pool(Tensor, Tensor, float, int)"/> for one image and tag on the CPU, from its real windows' raw dots.</summary>
    public static double Pool(IReadOnlyList<double> dots, float temperature, int topK, float bottomWeight = 0)
    {
        if (topK > 0)
            return dots.OrderDescending().Take(topK).Average() + (bottomWeight > 0 ? bottomWeight * dots.Order().Take(topK).Average() : 0);
        var max = dots.Max(d => d / temperature);
        var total = dots.Sum(d => Math.Exp(d / temperature - max));
        return temperature * (max + Math.Log(total) - Math.Log(dots.Count));
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
    public static Tensor TrainedRows(Tensor cells, Tensor valid, Tensor tagEmbeddings, long[] pooledRows, Training.PairTargets targets, float temperature, int minedPerImage, int topK = 0, float bottomWeight = 0)
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
                var dots = cells.transpose(1, 2).matmul(tagEmbeddings.index_select(0, rows).t()); // [B, K, L]
                var part = Pool(dots.transpose(1, 2), valid, temperature, topK, bottomWeight).cpu().data<float>().ToArray();
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
    /// tag, replaced for <paramref name="pooledRows"/> by <see cref="Pool(Tensor, Tensor, float, int)"/> over the valid cells.
    /// </summary>
    /// <param name="windowMask">[B, L, K] optional per-tag window choice (training's framed-positive pooling), within <paramref name="valid"/>.</param>
    public static Tensor RawScores(Tensor pooled, Tensor cells, Tensor valid, Tensor tagEmbeddings, Tensor pooledRows, float temperature, int topK = 0, float bottomWeight = 0, Tensor? windowMask = null)
    {
        var mean = pooled.matmul(tagEmbeddings.t());
        if (pooledRows.numel() == 0)
            return mean;
        var dots = cells.transpose(1, 2).matmul(tagEmbeddings.index_select(0, pooledRows).t()); // [B, K, L]
        var count = valid.sum(1, keepdim: true).to_type(ScalarType.Float32);
        var pooledScores = Pool(dots.transpose(1, 2), windowMask ?? valid, temperature, topK, bottomWeight);
        // An image with no cells at all (no glimpse of it) keeps its averaged score instead of -infinity.
        pooledScores = where(count.gt(0), pooledScores, mean.index_select(1, pooledRows));
        return mean.index_copy(1, pooledRows, pooledScores);
    }
}
