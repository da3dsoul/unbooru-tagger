using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Window labels from the model's own localization map (<c>--self-map-weight</c>, 2026-11-03; plan #1 of
/// review/open-vocab-research/PLAN.md, after Detic and OWL-ST: a region classifier learns from a localizer's regions, never from
/// image labels assigned to its own best region). For up to <see cref="Positives"/> labelled localizable positives per image,
/// the tag's map on the current step (detached) gives:
/// <list type="bullet">
/// <item>"yes": the 64px and 128px windows centred on the map's peak;</item>
/// <item>"no": <see cref="NoWindows"/> windows drawn from the image's 64px windows (stride 32px, inside the content) whose mean map
/// response is in its bottom <see cref="NoQuantile"/>.</item>
/// </list>
/// Measured on gold at run 4's 1.507M (<c>evaluate --map-window-check</c>): the peak window holds the object 70% of the time
/// (medium 72%), a bottom-40% window touches it 8% of the time (medium 5%, small 3%) against 18% for any window. Each window is
/// embedded by the image tower as a glimpse (<see cref="Glimpses.Embed"/>, 64px views, no zoom codes needed) and scored against
/// its tag; the loss is a per-window BCE. The image-level losses don't change; nothing here trains on negative images (SAM 3's
/// ablation: region scores shouldn't learn from image-level negatives).
/// </summary>
public static class SelfMapWindows
{
    public const int Positives = 4;
    public const int NoWindows = 2;
    public const double NoQuantile = 0.4;
    private static readonly int[] YesSides = [64, 128];
    private const int NoSide = 64, GridStride = 32;

    /// <summary>Windows per positive: the yes windows, then the no windows.</summary>
    public static int PerPositive => YesSides.Length + NoWindows;

    /// <summary>
    /// The windows (one <see cref="GlimpseChoice"/> of [B, P·<see cref="PerPositive"/>]) and their labels ([B, P·PerPositive], 1 = yes),
    /// on the device, without gradients and without reading anything back.
    /// </summary>
    /// <param name="tags">[B, P, D] the positives' tag embeddings (detached is fine), padded; <paramref name="valid"/> [B, P].</param>
    public static (GlimpseChoice Choice, Tensor Labels) Windows(Tensor localizationMap, Tensor mapMask, Tensor tags, Tensor valid, Tensor contentBoxes,
        int inputSize, int outputStride, Generator? generator = null)
    {
        using var _ = no_grad();
        using var scope = NewDisposeScope();
        var device = localizationMap.device;
        var (batch, positives) = (tags.shape[0], tags.shape[1]);
        var map = localizationMap.detach();
        var (h, w) = (map.shape[2], map.shape[3]);
        var heat = tags.detach().to_type(map.dtype).bmm(map.flatten(2)).view(batch, positives, h, w); // [B, P, h, w]
        var mask = mapMask.expand(-1, positives, -1, -1); // [B, P, h, w]

        // The peak (content only), as Glimpses.Peaks finds it.
        var peak = heat.masked_fill(mask.eq(0), float.NegativeInfinity).flatten(2).argmax(2); // [B, P]
        var peakX = peak.remainder(w) * outputStride + outputStride / 2;
        var peakY = peak.div(w, rounding_mode: RoundingMode.floor) * outputStride + outputStride / 2;

        var box = contentBoxes.unsqueeze(1); // [B, 1, 4]
        var (cx, cy, cw, ch) = (box[.., .., 0], box[.., .., 1], box[.., .., 2], box[.., .., 3]);
        Tensor Clamp(Tensor v, Tensor lo, Tensor extent, int side) =>
            minimum(maximum(v, lo), lo + extent - side).clamp(0, inputSize - side);

        var zooms = new List<Tensor>();
        var xs = new List<Tensor>();
        var ys = new List<Tensor>();
        var oks = new List<Tensor>();
        var labels = new List<Tensor>();
        foreach (var side in YesSides)
        {
            zooms.Add(full([batch, positives], ZoomOf(side), ScalarType.Int64, device: device));
            xs.Add(Clamp(peakX - side / 2, cx, cw, side));
            ys.Add(Clamp(peakY - side / 2, cy, ch, side));
            oks.Add(valid.logical_and(cw.ge(side)).logical_and(ch.ge(side)));
            labels.Add(ones([batch, positives], device: device));
        }

        // "No": the 64px grid windows inside the content, ranked by their mean response; drawn at random from the bottom share.
        var cells = NoSide / outputStride;
        var stride = GridStride / outputStride;
        var flatHeat = (heat * mask).view(batch * positives, 1, h, w);
        var flatMask = mask.reshape(batch * positives, 1, h, w).to_type(heat.dtype);
        var sums = nn.functional.avg_pool2d(flatHeat, cells, stride);
        var coverage = nn.functional.avg_pool2d(flatMask, cells, stride);
        var (gh, gw) = (sums.shape[2], sums.shape[3]);
        var means = (sums / coverage.clamp_min(1e-6f)).view(batch, positives, gh * gw);
        var inside = coverage.ge(0.999f).view(batch, positives, gh * gw);
        var sorted = means.masked_fill(inside.logical_not(), float.PositiveInfinity).sort(2).Values;
        var count = inside.sum(2); // [B, P]
        var at = (count.to_type(ScalarType.Float32) * (float)NoQuantile).floor().to_type(ScalarType.Int64).sub(1).clamp_min(0);
        var threshold = sorted.gather(2, at.unsqueeze(2)); // [B, P, 1]
        var eligible = inside.logical_and(means.le(threshold)).logical_and(count.ge(5).unsqueeze(2));
        var draw = rand([batch, positives, gh * gw], generator: generator, device: device).masked_fill(eligible.logical_not(), -1f);
        var picked = draw.topk(NoWindows, dim: 2);
        for (var k = 0; k < NoWindows; k++)
        {
            var index = picked.indexes.select(2, k);
            zooms.Add(full([batch, positives], ZoomOf(NoSide), ScalarType.Int64, device: device));
            xs.Add(index.remainder(gw) * GridStride);
            ys.Add(index.div(gw, rounding_mode: RoundingMode.floor) * GridStride);
            oks.Add(valid.logical_and(picked.values.select(2, k).ge(0)));
            labels.Add(zeros([batch, positives], device: device));
        }

        // [B, P, PerPositive] -> [B, P·PerPositive], positive-major.
        Tensor Stack(List<Tensor> parts) => stack(parts, 2).reshape(batch, positives * PerPositive);
        var choice = new GlimpseChoice(Stack(zooms), Stack(xs), Stack(ys), Stack(oks));
        var result = (choice, Stack(labels));
        scope.MoveToOuter(choice.Zoom, choice.X, choice.Y, choice.Valid, result.Item2);
        return result;
    }

    /// <summary>Mean BCE over the real windows of each window's score for its own tag.</summary>
    /// <param name="cells">[B, D, P·PerPositive] the windows' embeddings (with gradient).</param>
    /// <param name="tags">[B, P, D] the tags (with gradient).</param>
    public static Tensor Term(Tensor cells, Tensor tags, Tensor labels, Tensor windowValid, LogitScaleBias scaling, bool detachScaling = false)
    {
        using var scope = NewDisposeScope();
        var (batch, positives) = (tags.shape[0], tags.shape[1]);
        var perWindow = tags.to_type(cells.dtype).unsqueeze(2).expand(-1, -1, PerPositive, -1).reshape(batch, positives * PerPositive, -1); // [B, K, D]
        var dots = (perWindow * cells.transpose(1, 2)).sum(2); // [B, K]
        var logits = detachScaling ? scaling.ForwardDetached(dots) : scaling.forward(dots);
        var loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction: nn.Reduction.None);
        var real = windowValid.to_type(loss.dtype);
        return ((loss * real).sum() / real.sum().clamp_min(1f)).MoveToOuterDisposeScope();
    }

    /// <summary>[1, slots·PerPositive]: true on each slot's peak (yes-side) windows, false on its bottom windows.</summary>
    public static Tensor PeakWindowMask(int slots, Device device) =>
        tensor(Enumerable.Range(0, slots * PerPositive).Select(k => k % PerPositive < YesSides.Length).ToArray(), [1, slots * PerPositive]).to(device);

    private static long ZoomOf(int side) => Array.FindIndex(Enumerable.Range(0, Glimpses.Zooms.Length).ToArray(), z => Glimpses.SideOf(z) == side);
}
