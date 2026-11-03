using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Matched twins' window labels (<c>--twin-every</c>, <see cref="TagCentricBatchSampler.SampleTwins"/>; the user's idea, 2026-11-02):
/// a batch of one image with a tag (gloves) and twins without it whose tags otherwise overlap it most. A glimpse of the positive
/// that looks like glimpses of several twins shows something they share (her face, her hair, the same coat), not the tag, so it is
/// trained "no" for the tag, window by window: the within-image "no" the gold-box ceiling test had (a 64px glimpse then told the
/// object from other objects at 0.80 AUC) and tags alone, or mosaics, can't give. There is no "yes" side here: the positive's
/// ordinary glimpse loss already says the tag is somewhere, and mosaics' tile "yes" bags raised every score (mg1).
/// <list type="bullet">
/// <item>Similarity is the cosine between glimpse embeddings, detached: a positive window's match to a twin is its best match
/// among that twin's windows, and its score the <see cref="Matches"/>th best twin's, so one twin with the tag untagged
/// (~3% of negatives) can't make the object's own window look shared.</item>
/// <item>The most twin-like half of the positive's windows are labelled "no", leaving out any window holding the tag's own map
/// peak, where the model already thinks the tag is.</item>
/// </list>
/// </summary>
public static class TwinWindows
{
    /// <summary>How many twins a window must match: its score is its <c>Matches</c>th best twin match.</summary>
    public const int Matches = 3;

    /// <summary>
    /// [K] which of the positive's windows are labelled "no", on the device, without gradients.
    /// </summary>
    /// <param name="cells">[B, D, K] the batch's glimpse embeddings.</param>
    /// <param name="valid">[B, K] which windows are real.</param>
    /// <param name="twins">[N] the twins' batch rows, padded to a fixed count (<paramref name="twinValid"/> false).</param>
    /// <param name="peakX">The tag's map peak on the positive, canvas x ([1] or scalar).</param>
    public static Tensor NoWindows(Tensor cells, Tensor valid, long positive, Tensor twins, Tensor twinValid, GlimpseChoice choice, Tensor peakX, Tensor peakY)
    {
        using var _ = no_grad();
        using var scope = NewDisposeScope();
        var normalized = nn.functional.normalize(cells.detach().to_type(ScalarType.Float32), dim: 1); // [B, D, K]
        var own = normalized[positive].t().unsqueeze(0); // [1, K, D]
        var others = normalized.index_select(0, twins); // [N, D, K']
        var similarity = own.expand(others.shape[0], -1, -1).bmm(others); // [N, K, K']
        var otherValid = valid.index_select(0, twins).unsqueeze(1); // [N, 1, K']
        var best = where(otherValid, similarity, full_like(similarity, -2f)).amax(2); // [N, K]
        best = where(twinValid.unsqueeze(1), best, full_like(best, -2f));
        var matches = Math.Min(Matches, (int)twins.shape[0]);
        var score = best.topk(matches, 0).values[matches - 1]; // [K]

        var side = tensor(Enumerable.Range(0, Glimpses.Zooms.Length).Select(z => (long)Glimpses.SideOf(z)).ToArray()).to(cells.device)
            .index_select(0, choice.Zoom[positive]);
        var (x, y) = (choice.X[positive], choice.Y[positive]);
        var (px, py) = (peakX.reshape(1), peakY.reshape(1));
        var holdsPeak = x.le(px).logical_and(px.lt(x + side)).logical_and(y.le(py)).logical_and(py.lt(y + side));
        var eligible = valid[positive].logical_and(holdsPeak.logical_not());

        // The most twin-like half of the eligible windows: at or above the median eligible score.
        var sorted = where(eligible, score, full_like(score, float.NegativeInfinity)).sort(descending: true).Values;
        var middle = ((eligible.sum() - 1) / 2).clamp_min(0).to_type(ScalarType.Int64).reshape(1);
        var threshold = sorted.gather(0, middle);
        return eligible.logical_and(score.ge(threshold)).MoveToOuterDisposeScope();
    }

    /// <summary>Mean BCE of "no" for the tag on the positive's <paramref name="no"/> windows.</summary>
    /// <param name="tag">[D] the tag's embedding (with gradient).</param>
    /// <param name="detachScaling">No gradient into the shared logit scale and bias (<c>--twin-detach-scaling</c>).</param>
    public static Tensor Term(Tensor cells, long positive, Tensor tag, Tensor no, LogitScaleBias scaling, bool detachScaling = false)
    {
        using var scope = NewDisposeScope();
        var dots = tag.unsqueeze(0).matmul(cells[positive].to_type(tag.dtype)).squeeze(0); // [K]
        var logits = detachScaling ? scaling.ForwardDetached(dots) : scaling.forward(dots);
        var loss = nn.functional.binary_cross_entropy_with_logits(logits, zeros_like(logits), reduction: nn.Reduction.None);
        return ((loss * no).sum() / no.sum().clamp_min(1)).MoveToOuterDisposeScope();
    }
}
