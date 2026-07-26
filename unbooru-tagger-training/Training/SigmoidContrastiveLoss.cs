using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// SigLIP-style sigmoid loss: every (image, tag) pair in the batch is an independent
/// binary match/no-match decision, not a softmax over the batch (CLAUDE.md's training
/// objective — confidence for a pair is sigmoid(dot(image, tag))).
/// </summary>
public static class SigmoidContrastiveLoss
{
    /// <param name="imageEmbeddings">[batchImages, dim]</param>
    /// <param name="tagEmbeddings">[batchTags, dim]</param>
    /// <param name="labels">[batchImages, batchTags]: +1 where the image is tagged with that tag, -1 otherwise.</param>
    public static Tensor Compute(Tensor imageEmbeddings, Tensor tagEmbeddings, Tensor labels)
    {
        var logits = imageEmbeddings.matmul(tagEmbeddings.t());
        // -log(sigmoid(labels * logits)) == softplus(-labels * logits): the standard
        // binary-cross-entropy-with-logits form of the sigmoid loss.
        return nn.functional.softplus(-labels * logits).mean();
    }

    /// <summary>
    /// Same sigmoid loss as <see cref="Compute"/>, but from per-location logits instead of
    /// one pre-pooled image embedding — a multiple-instance-learning objective that directly
    /// rewards sharp spatial responses instead of leaving localization as a side effect of
    /// the dual-encoder geometry (CLAUDE.md's localization section). Per-location logits are
    /// combined with a temperature-controlled log-sum-exp instead of a hard max, which spans
    /// average pooling (<paramref name="temperature"/> -> infinity, mathematically identical
    /// to <see cref="Compute"/> since dot product distributes over an average) down to max
    /// pooling (<paramref name="temperature"/> -> 0) as temperature drops.
    /// </summary>
    /// <param name="spatialFeatures">[batchImages, embeddingDim, height, width] — the image tower's pre-pool spatial map.</param>
    /// <param name="tagEmbeddings">[batchTags, embeddingDim]</param>
    /// <param name="labels">[batchImages, batchTags]: +1 where the image is tagged with that tag, -1 otherwise.</param>
    /// <param name="temperature">Lower sharpens (concentrates gradient on a tag's best-matching location); higher smooths toward <see cref="Compute"/>'s behavior.</param>
    /// <param name="spatialMask">
    /// Optional [batchImages, 1, height, width] validity mask (see
    /// <c>UnbooruTagger.Training.Model.SpatialMask</c>) — 1 for locations with real image
    /// content, 0 for letterbox padding. Masked-out locations are excluded from both the
    /// log-sum-exp pool and the log-location-count normalization it's built on, so a
    /// padded border can't be picked as a tag's "best-matching location" or dilute the
    /// mean-pooled fallback at high temperature. Omit for an unpadded spatial map (every
    /// location counts, matching the original unmasked behavior).
    /// </param>
    public static Tensor ComputeLocalized(Tensor spatialFeatures, Tensor tagEmbeddings, Tensor labels, float temperature, Tensor? spatialMask = null)
    {
        var batch = spatialFeatures.shape[0];
        var dim = spatialFeatures.shape[1];
        var height = spatialFeatures.shape[2];
        var width = spatialFeatures.shape[3];
        var locations = height * width;

        // [B, D, H, W] -> [B, L, D] so a batched matmul against tagEmbeddings.T scores every
        // location against every tag at once: [B, L, D] x [D, T] -> [B, L, T] -> [B, T, L].
        var flattened = spatialFeatures.reshape([batch, dim, locations]).transpose(1, 2);
        var perLocationLogits = flattened.matmul(tagEmbeddings.t()).transpose(1, 2);

        var scaled = perLocationLogits / temperature;

        Tensor logValidCount;
        if (spatialMask is not null)
        {
            var maskFlat = spatialMask.reshape([batch, 1, locations]);
            scaled = scaled.masked_fill(maskFlat.eq(0), float.NegativeInfinity);
            logValidCount = maskFlat.sum([2], keepdim: true).log();
        }
        else
        {
            logValidCount = tensor(MathF.Log(locations));
        }

        var maxScaled = scaled.amax([2], keepdim: true);
        var logSumExp = maxScaled + (scaled - maxScaled).exp().sum([2], keepdim: true).log();
        var pooledLogits = (temperature * (logSumExp - logValidCount)).squeeze(2);

        return nn.functional.softplus(-labels * pooledLogits).mean();
    }
}
