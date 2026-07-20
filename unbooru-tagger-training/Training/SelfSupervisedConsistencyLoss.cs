using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// SimSiam-style self-supervised consistency loss (Chen &amp; He, 2021): two independently
/// augmented views of the same image (see <see cref="RandomCropAugmentation"/>) should
/// predict each other's pooled embedding. Needs no tag labels at all, so it's free training
/// signal from images already in the corpus — pushes the image tower toward representations
/// that are stable under crop/flip, which (CLAUDE.md's localization section) tends to sharpen
/// what the spatial feature map actually responds to, instead of scattering weight across
/// whatever happened to survive one particular crop.
/// </summary>
public static class SelfSupervisedConsistencyLoss
{
    /// <param name="pooledA">pooled(viewA) — the stop-gradient target for <paramref name="predictionB"/>.</param>
    /// <param name="predictionA">predictionHead(pooledA)</param>
    /// <param name="pooledB">pooled(viewB) — the stop-gradient target for <paramref name="predictionA"/>.</param>
    /// <param name="predictionB">predictionHead(pooledB)</param>
    public static Tensor Compute(Tensor pooledA, Tensor predictionA, Tensor pooledB, Tensor predictionB)
    {
        using var targetA = pooledA.detach();
        using var targetB = pooledB.detach();

        var lossA = NegativeCosineSimilarity(predictionA, targetB);
        var lossB = NegativeCosineSimilarity(predictionB, targetA);
        return (lossA + lossB) / 2;
    }

    private static Tensor NegativeCosineSimilarity(Tensor prediction, Tensor target)
    {
        var predictionNorm = prediction / (prediction.pow(2).sum([1], keepdim: true).sqrt() + 1e-8);
        var targetNorm = target / (target.pow(2).sum([1], keepdim: true).sqrt() + 1e-8);
        return -(predictionNorm * targetNorm).sum([1]).mean();
    }
}
