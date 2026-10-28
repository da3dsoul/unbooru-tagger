using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// SigLIP-style sigmoid loss: every (image, tag) pair in the batch is an independent
/// binary match/no-match decision, not a softmax over the batch (CLAUDE.md's training
/// objective — confidence for a pair is sigmoid(scale * dot(image, tag) + bias)).
/// </summary>
public static class SigmoidContrastiveLoss
{
    /// <param name="imageEmbeddings">[batchImages, dim]</param>
    /// <param name="tagEmbeddings">[batchTags, dim]</param>
    /// <param name="labels">[batchImages, batchTags]: +1 where the image is tagged with that tag, -1 otherwise.</param>
    /// <param name="scaling">
    /// The learnable <see cref="LogitScaleBias"/> applied before the sigmoid (see its own
    /// doc comment for why). Omit for the raw, unscaled/unbiased dot product — the
    /// pre-<see cref="LogitScaleBias"/> behavior every caller had before it existed.
    /// </param>
    public static Tensor Compute(Tensor imageEmbeddings, Tensor tagEmbeddings, Tensor labels, LogitScaleBias? scaling = null)
    {
        var rawLogits = imageEmbeddings.matmul(tagEmbeddings.t());
        var logits = scaling?.forward(rawLogits) ?? rawLogits;
        // -log(sigmoid(labels * logits)) == softplus(-labels * logits): the standard
        // binary-cross-entropy-with-logits form of the sigmoid loss.
        return nn.functional.softplus(-labels * logits).mean();
    }

    /// <summary>
    /// The deliberate-targets version of <see cref="Compute"/>: every pair carries its own
    /// weight and certainty (<see cref="PairWeightBuilder"/>), and unverified negatives use
    /// the asymmetric loss (ASL, Ridnik et al. 2021) instead of plain BCE — their predicted
    /// probability is shifted down by <paramref name="negativeClip"/> (anything below it
    /// costs exactly nothing) and the rest is damped by <c>p^gamma</c>, so the ~25K
    /// unverified negatives per image stop drowning out its ~40 positives, and a tag the
    /// tagger merely forgot costs little. Positives and certain negatives (anchor hard
    /// negatives, mutually-exclusive tags) keep full-strength BCE: those are exactly the
    /// distinctions the model has to get right.
    ///
    /// Normalized by the batch's total positive weight rather than by every pair: with
    /// almost every pair zero-weight or focal-damped, a per-pair mean would shrink the
    /// gradient toward nothing as the vocabulary grows.
    ///
    /// <see cref="PairTargets.LogitOffsets"/>, if any, are added to the logits here (<see cref="ContextPrior"/>).
    /// </summary>
    public static Tensor ComputeWeighted(
        Tensor imageEmbeddings, Tensor tagEmbeddings, PairTargets targets, LogitScaleBias? scaling,
        float negativeGamma, float negativeClip)
    {
        return ComputeWeightedScores(imageEmbeddings.matmul(tagEmbeddings.t()), targets, scaling, negativeGamma, negativeClip);
    }

    /// <summary><see cref="ComputeWeighted"/> from [batch, vocabulary] raw scores (before <paramref name="scaling"/>), for scores not from one pooled embedding (<see cref="Model.ClassificationPooling"/>).</summary>
    public static Tensor ComputeWeightedScores(Tensor rawLogits, PairTargets targets, LogitScaleBias? scaling, float negativeGamma, float negativeClip)
    {
        var logits = scaling?.forward(rawLogits) ?? rawLogits;
        if (targets.LogitOffsets is not null)
            logits = logits + targets.LogitOffsets;
        return PairLoss(logits, targets.Labels, targets.Weights, targets.FocalMask, negativeGamma, negativeClip).sum()
            / Math.Max(1f, targets.PositiveWeightSum);
    }

    /// <summary>
    /// The per-location (MIL/log-sum-exp) loss of <see cref="ComputeLocalized"/>, computed only
    /// on <see cref="LocalizationPairSelector"/>'s chosen pairs instead of every tag in the
    /// vocabulary: one [batch, pairs, locations] batched matmul (pairs ≈ 100) in place of
    /// [batch, vocabulary, locations] (vocabulary ≈ 25K). Same pooling, scaling and
    /// weighted/asymmetric per-pair loss as <see cref="ComputeWeighted"/>; returns a tensor to
    /// fold into the step's single loss, since it's small enough not to need
    /// <see cref="ComputeLocalizedChunked"/>'s per-chunk backward.
    /// <para>
    /// <paramref name="backgroundWeight"/> &gt; 0 adds background suppression. For every
    /// positive pair, the <paramref name="backgroundFraction"/> of the image's valid locations
    /// that score lowest for that tag (ranked on detached scores) are pooled the same way and
    /// trained as a negative. The MIL term alone only asks that some location match, and the
    /// cheapest way for a large-receptive-field tower to satisfy it is to score every location
    /// high: run 2's epoch-1 heatmaps were near 1.0 across the whole image. Taking the
    /// least tag-like locations leaves extra instances alone (a second necktie still ranks
    /// high), so it doesn't fight multi-instance tags.
    /// </para>
    /// </summary>
    public static Tensor ComputeLocalizedPairs(
        Tensor spatialFeatures, Tensor tagEmbeddings, LocalizationPairs pairs, float temperature,
        Tensor? spatialMask, LogitScaleBias? scaling, float negativeGamma, float negativeClip,
        float backgroundWeight = 0f, float backgroundFraction = 0.4f)
    {
        var batch = spatialFeatures.shape[0];
        var dim = spatialFeatures.shape[1];
        var locations = spatialFeatures.shape[2] * spatialFeatures.shape[3];
        var width = pairs.TagRows.shape[1];

        var pairEmbeddings = tagEmbeddings.index_select(0, pairs.TagRows.flatten()).reshape([batch, width, dim]);
        var perLocation = pairEmbeddings.bmm(spatialFeatures.reshape([batch, dim, locations])); // [B, K, L]

        var scaled = perLocation / temperature;
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
        var rawPooled = (temperature * (logSumExp - logValidCount)).squeeze(2);
        var pooled = scaling?.forward(rawPooled) ?? rawPooled;

        var loss = PairLoss(pooled, pairs.Labels, pairs.Weights, pairs.FocalMask, negativeGamma, negativeClip).sum();
        if (backgroundWeight > 0)
            loss = loss + backgroundWeight * BackgroundLoss(perLocation, pairs, temperature, spatialMask, scaling, backgroundFraction);
        return loss / Math.Max(1f, pairs.PositiveWeightSum);
    }

    /// <summary>
    /// Crop consistency: the classifier teaches localization. <paramref name="teacherLogits"/>
    /// ([B, K], computed without gradients) are the whole-image classifier's logits for each
    /// pair's tag on a crop of the image (<see cref="CropConsistency"/>), i.e. whether the tag
    /// is inside that window. The full image's localization map, pooled over the same window
    /// the way the MIL loss pools the whole map, is trained toward that probability (soft
    /// BCE), weighted by the teacher's confidence |2p - 1| so a crop that cuts an object in
    /// half teaches little. Only positive pairs count: a heatmap peak sitting on background is
    /// pulled off it because the classifier, shown that background alone, doesn't find the tag.
    /// </summary>
    /// <param name="windowMask">[B, 1, H, W], 1 inside each image's crop window.</param>
    /// <param name="pairMask">Optional [B, K] 0/1: which pairs' tags the teacher is trusted on (CropTeacherGate).</param>
    /// <param name="confidentAbove">&gt; 0: train only on teacher answers with p above this or below 1 minus it, unweighted, instead of weighting every answer by |2p - 1|.</param>
    public static Tensor ComputeWindowConsistency(
        Tensor localizationMap, Tensor tagEmbeddings, LocalizationPairs pairs, Tensor windowMask, Tensor teacherLogits,
        float temperature, LogitScaleBias? scaling, Tensor? pairMask = null, float confidentAbove = 0f)
    {
        var batch = localizationMap.shape[0];
        var dim = localizationMap.shape[1];
        var locations = localizationMap.shape[2] * localizationMap.shape[3];
        var width = pairs.TagRows.shape[1];

        var pairEmbeddings = tagEmbeddings.index_select(0, pairs.TagRows.flatten()).reshape([batch, width, dim]);
        var perLocation = pairEmbeddings.bmm(localizationMap.reshape([batch, dim, locations])); // [B, K, L]

        var inside = windowMask.reshape([batch, 1, locations]).gt(0);
        var scaled = (perLocation / temperature).masked_fill(inside.logical_not(), float.NegativeInfinity);
        var maxScaled = scaled.amax([2], keepdim: true).detach();
        var logSumExp = maxScaled + (scaled - maxScaled).exp().sum([2], keepdim: true).log();
        var count = inside.sum([2], keepdim: true).to_type(ScalarType.Float32).clamp_min(1);
        var rawPooled = (temperature * (logSumExp - count.log())).squeeze(2);
        var student = scaling?.forward(rawPooled) ?? rawPooled;

        var target = teacherLogits.detach().sigmoid();
        var confidence = confidentAbove > 0
            ? target.gt(confidentAbove).logical_or(target.lt(1 - confidentAbove)).to_type(student.dtype)
            : (2 * target - 1).abs();
        if (pairMask is not null)
            confidence = confidence * pairMask;
        var isPositive = pairs.Labels.gt(0).logical_and(pairs.Weights.gt(0)).to_type(student.dtype);
        var bce = nn.functional.softplus(student) - target * student;
        return (bce * confidence * pairs.Weights * isPositive).sum() / Math.Max(1f, pairs.PositiveWeightSum);
    }

    /// <summary>Summed (not normalized) background-suppression loss — see <see cref="ComputeLocalizedPairs"/>.</summary>
    private static Tensor BackgroundLoss(
        Tensor perLocation, LocalizationPairs pairs, float temperature, Tensor? spatialMask, LogitScaleBias? scaling, float fraction)
    {
        var batch = perLocation.shape[0];
        var width = perLocation.shape[1];
        var locations = perLocation.shape[2];
        var valid = spatialMask is not null
            ? spatialMask.reshape([batch, 1, locations]).gt(0).expand([batch, width, locations])
            : ones([batch, width, locations], ScalarType.Bool, perLocation.device);

        Tensor background;
        using (no_grad())
        {
            // The k-th lowest valid score per pair is the cutoff; invalid (padding) locations
            // sort last so they never count.
            var ranked = perLocation.detach().masked_fill(valid.logical_not(), float.PositiveInfinity);
            var sorted = ranked.sort(2).Values;
            var validCount = valid.sum([2], keepdim: true);
            var k = (validCount.to_type(ScalarType.Float32) * fraction).floor().to_type(ScalarType.Int64).clamp_min(1) - 1;
            var cutoff = sorted.gather(2, k);
            background = valid.logical_and(ranked.le(cutoff));
        }

        var scaled = (perLocation / temperature).masked_fill(background.logical_not(), float.NegativeInfinity);
        var maxScaled = scaled.amax([2], keepdim: true).detach();
        var logSumExp = maxScaled + (scaled - maxScaled).exp().sum([2], keepdim: true).log();
        var count = background.sum([2], keepdim: true).to_type(ScalarType.Float32).clamp_min(1);
        var rawPooled = (temperature * (logSumExp - count.log())).squeeze(2);
        var pooled = scaling?.forward(rawPooled) ?? rawPooled;

        var isPositive = pairs.Labels.gt(0).logical_and(pairs.Weights.gt(0)).to_type(pooled.dtype);
        return (nn.functional.softplus(pooled) * pairs.Weights * isPositive).sum();
    }

    /// <summary>Per-pair weighted loss shared by <see cref="ComputeWeighted"/> and <see cref="ComputeLocalizedChunked"/> — see <see cref="ComputeWeighted"/>.</summary>
    private static Tensor PairLoss(Tensor logits, Tensor labels, Tensor weights, Tensor focalMask, float negativeGamma, float negativeClip)
    {
        var bce = nn.functional.softplus(-labels * logits);

        // ASL negative term: -(p_m)^gamma * log(1 - p_m), p_m = max(p - clip, 0). With
        // clip > 0, 1 - p_m >= clip, so the log is always finite; with clip == 0 it's
        // exactly softplus(logits), computed stably that way instead. The focal factor is
        // detached (treated as a per-pair weight), the common stable ASL variant.
        Tensor asl;
        if (negativeClip > 0)
        {
            var shifted = (logits.sigmoid() - negativeClip).clamp_min(0);
            asl = -(-shifted).log1p();
            if (negativeGamma > 0)
                asl = asl * shifted.detach().pow(negativeGamma);
        }
        else
        {
            asl = nn.functional.softplus(logits);
            if (negativeGamma > 0)
                asl = asl * logits.sigmoid().detach().pow(negativeGamma);
        }

        return where(focalMask > 0, asl, bce) * weights;
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
    /// <param name="scaling">Same <see cref="LogitScaleBias"/> as <see cref="Compute"/>, applied after pooling so the high-temperature equivalence with <see cref="Compute"/> still holds exactly.</param>
    public static Tensor ComputeLocalized(Tensor spatialFeatures, Tensor tagEmbeddings, Tensor labels, float temperature, Tensor? spatialMask = null, LogitScaleBias? scaling = null)
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
        var rawPooledLogits = (temperature * (logSumExp - logValidCount)).squeeze(2);
        var pooledLogits = scaling?.forward(rawPooledLogits) ?? rawPooledLogits;

        return nn.functional.softplus(-labels * pooledLogits).mean();
    }

    /// <summary>
    /// Same objective as <see cref="ComputeLocalized"/>, computed in chunks along the
    /// tag dimension instead of scoring the whole vocabulary against every location in
    /// one <c>[batch, vocabSize, locations]</c> tensor — the memory-dominant tensor at
    /// this vocabulary's scale, since it's the only one of the loss terms whose size
    /// scales with <c>locations</c> as well as vocabulary size. Exact, not an
    /// approximation: each tag's log-sum-exp pooling only ever depends on its own
    /// locations, never other tags, so splitting the vocabulary and recombining is just
    /// a different order of the same sum (<c>ComputeLocalizedChunked_ProducesTheSameGradientsAsTheUnchunkedVersion</c>
    /// in the test suite verifies this directly, by gradient comparison against
    /// <see cref="ComputeLocalized"/>, not just by argument).
    ///
    /// Unlike every other method here, this does NOT return a <see cref="Tensor"/> for
    /// the caller to fold into one combined loss and back-propagate once. It calls
    /// <c>backward(retain_graph: true)</c> itself, once per chunk, immediately after
    /// that chunk's forward pass. That's the whole point: a single backward() call at
    /// the end — no matter how the forward pass was chunked — can't reduce peak memory,
    /// because autograd keeps every executed op's saved-for-backward tensors alive
    /// until backward() actually consumes them, so all chunks would still coexist right
    /// up to that point. Backpropagating chunk-by-chunk means each chunk's tensors can
    /// be freed (the per-chunk <c>NewDisposeScope()</c> below) before the next chunk's
    /// forward pass even starts.
    ///
    /// <paramref name="scaling"/>'s parameters and <paramref name="tagEmbeddings"/>' and
    /// <paramref name="spatialFeatures"/>' own upstream graphs (e.g. back through
    /// <c>TagTower</c>/<c>ImageTower</c>) receive gradients exactly as any other
    /// backward() call would — <c>retain_graph: true</c> only keeps the graph alive for
    /// further backward calls the training step still has left to make (e.g. the
    /// primary <see cref="Compute"/> loss, which shares this same upstream graph and
    /// must be computed and backpropagated AFTER this method returns, not before — see
    /// <see cref="Commands.TrainCommandHandler"/>).
    /// </summary>
    /// <param name="tagChunkSize">Tags processed per chunk. Smaller bounds peak memory tighter at the cost of more, smaller matmuls (some fixed per-chunk overhead) — tune to whatever fits.</param>
    /// <param name="gradientScale">
    /// Multiplies every chunk's gradient contribution — this is how <see cref="Commands.TrainCommandHandler"/>
    /// folds in both <c>--localization-weight</c> and (since this bypasses the normal
    /// "build one loss tensor, divide by --gradient-accumulation-steps, backward once"
    /// path) the <c>1 / gradientAccumulationSteps</c> scale gradient accumulation needs.
    /// </param>
    /// <returns>The mean loss value (matching what <see cref="ComputeLocalized"/> would have returned), detached — for logging/NaN-checking only, already backpropagated.</returns>
    /// <param name="targets">
    /// Optional deliberate per-pair targets (see <see cref="ComputeWeighted"/>). When given,
    /// <paramref name="labels"/> is ignored in favor of <c>targets.Labels</c>, each pair's loss
    /// is <see cref="ComputeWeighted"/>'s weighted/asymmetric one, and the result is
    /// normalized by <c>targets.PositiveWeightSum</c> instead of by every pair.
    /// </param>
    public static float ComputeLocalizedChunked(
        Tensor spatialFeatures, Tensor tagEmbeddings, Tensor labels, float temperature,
        int tagChunkSize, float gradientScale, Tensor? spatialMask = null, LogitScaleBias? scaling = null,
        PairTargets? targets = null, float negativeGamma = 0f, float negativeClip = 0f)
    {
        var batch = spatialFeatures.shape[0];
        var dim = spatialFeatures.shape[1];
        var height = spatialFeatures.shape[2];
        var width = spatialFeatures.shape[3];
        var locations = height * width;
        var vocabSize = tagEmbeddings.shape[0];

        // Shared by every chunk -- built once, outside the chunk loop, so it isn't
        // recomputed (or accidentally freed) per chunk.
        var flattened = spatialFeatures.reshape([batch, dim, locations]).transpose(1, 2);

        Tensor? maskFlat = spatialMask?.reshape([batch, 1, locations]);
        var logValidCount = maskFlat is not null
            ? maskFlat.sum([2], keepdim: true).log()
            : tensor(MathF.Log(locations));

        var normalizer = targets is not null ? Math.Max(1f, targets.PositiveWeightSum) : batch * vocabSize;

        var totalLossSum = 0f;
        for (var start = 0; start < vocabSize; start += tagChunkSize)
        {
            var count = (int)Math.Min(tagChunkSize, vocabSize - start);

            // Frees this chunk's [batch, count, locations]-scale intermediates (perLocationLogits,
            // scaled, maxScaled, logSumExp, ...) once this iteration's backward() below has
            // used them -- see this method's own doc comment for why that's what actually
            // bounds peak memory here, not just the chunking itself.
            using var chunkScope = NewDisposeScope();

            var tagChunk = tagEmbeddings.narrow(0, start, count);
            var labelsChunk = labels.narrow(1, start, count);

            var perLocationLogits = flattened.matmul(tagChunk.t()).transpose(1, 2);
            var scaled = perLocationLogits / temperature;
            if (maskFlat is not null)
                scaled = scaled.masked_fill(maskFlat.eq(0), float.NegativeInfinity);

            var maxScaled = scaled.amax([2], keepdim: true);
            var logSumExp = maxScaled + (scaled - maxScaled).exp().sum([2], keepdim: true).log();
            var rawPooledLogits = (temperature * (logSumExp - logValidCount)).squeeze(2);
            var pooledLogits = scaling?.forward(rawPooledLogits) ?? rawPooledLogits;

            // Sum, not mean: chunks can differ in tag count (a trailing partial chunk),
            // so normalizing by the full (batch * vocabSize) count -- or the positive weight
            // sum, with deliberate targets -- happens once, after
            // every chunk's sum has been combined -- not per chunk, which would let a
            // short trailing chunk's tags implicitly outweigh a full chunk's.
            var chunkLossSum = targets is not null
                ? PairLoss(
                    pooledLogits, targets.Labels.narrow(1, start, count), targets.Weights.narrow(1, start, count),
                    targets.FocalMask.narrow(1, start, count), negativeGamma, negativeClip).sum()
                : nn.functional.softplus(-labelsChunk * pooledLogits).sum();
            totalLossSum += chunkLossSum.item<float>();

            (chunkLossSum * (gradientScale / normalizer)).backward(retain_graph: true);
        }

        return totalLossSum / normalizer;
    }
}
