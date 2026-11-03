using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Scoring;
using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Pasted twins (<c>--twin-paste</c>, 2026-11-02; with matched twins, <see cref="TagCentricBatchSampler.SampleTwins"/>): the
/// positive's region around its tag's localization-map peak is pasted into a twin at the same place, which then carries the tag;
/// a same-size region from elsewhere in the positive is pasted into another twin, which still doesn't. The two differ only in
/// what the patch shows, so a pasted-patch cue (the erased views learned "a grey patch lowers a score") can't pay, and the
/// tag's "yes" comes with near-identical context on both sides. Needs a map that points: run 4's peak lands on the held-out
/// gold-small objects 55% of the time (medium 67%), run 5's at 189K 8.5%. Pastes only when the peak itself is a "yes"
/// (the map's own scaled logit there at least 0), and blends the patch in over <see cref="Feather"/> pixels.
/// </summary>
public static class TwinPaste
{
    /// <summary>Patch sides, one drawn per paste.</summary>
    public static readonly int[] Sides = [64, 96, 128];

    /// <summary>Pixels over which a patch's edge blends into the twin.</summary>
    public const int Feather = 8;

    public sealed record Result(int[] Pasted, int[] Controls, bool Gated);

    /// <summary>
    /// Pastes into <paramref name="pixels"/> in place (no gradient): up to <paramref name="pairs"/> twins get the peak region and
    /// as many others a control region. Returns which batch rows now carry the tag and which are its controls.
    /// </summary>
    /// <param name="map">[1, D, h, w] the positive's localization map, from a no-gradient forward; <paramref name="mapMask"/> [1, 1, h, w].</param>
    /// <param name="tag">[D] the anchor tag's embedding.</param>
    public static Result Apply(Tensor pixels, IReadOnlyList<LetterboxBox> content, Tensor map, Tensor mapMask, Tensor tag, LogitScaling scaling,
        int positive, IReadOnlyList<int> twins, int pairs, int outputStride, Random random)
    {
        using var _ = no_grad();
        using var scope = NewDisposeScope();
        var heat = (map[0] * tag.to_type(map.dtype).view(-1, 1, 1)).sum(dim: 0); // [h, w] (sum(0) would bind to sum(ScalarType))
        heat = where(mapMask[0, 0].gt(0), heat, full_like(heat, float.NegativeInfinity));
        var width = heat.shape[1];
        var flat = heat.flatten();
        var index = flat.argmax().item<long>();
        var peak = flat[index].item<float>();
        if (scaling.Apply(peak) < 0 || twins.Count < 2)
            return new Result([], [], Gated: true);
        var (px, py) = ((int)((index % width + 0.5) * outputStride), (int)((index / width + 0.5) * outputStride));

        var count = Math.Min(pairs, twins.Count / 2);
        var pasted = new List<int>();
        var controls = new List<int>();
        var source = content[positive];
        for (var k = 0; k < count; k++)
        {
            var side = Sides[random.Next(Sides.Length)];
            var region = Square(px - side / 2, py - side / 2, side, source);
            if (Paste(pixels, positive, twins[2 * k], region, content[twins[2 * k]]))
                pasted.Add(twins[2 * k]);
            // The control: the same size from elsewhere in the positive, clear of the peak region.
            for (var attempt = 0; attempt < 20; attempt++)
            {
                var other = Square(source.X + random.Next(Math.Max(1, source.Width - side)), source.Y + random.Next(Math.Max(1, source.Height - side)), side, source);
                if (Overlaps(other, region))
                    continue;
                if (Paste(pixels, positive, twins[2 * k + 1], other, content[twins[2 * k + 1]]))
                    controls.Add(twins[2 * k + 1]);
                break;
            }
        }
        return new Result(pasted.ToArray(), controls.ToArray(), Gated: false);
    }

    /// <summary>A square of <paramref name="side"/> at (x, y), moved inside <paramref name="within"/> (shrunk if it doesn't fit).</summary>
    public static (int X, int Y, int Side) Square(int x, int y, int side, LetterboxBox within)
    {
        side = Math.Min(side, Math.Min(within.Width, within.Height));
        return (Math.Clamp(x, within.X, within.X + within.Width - side), Math.Clamp(y, within.Y, within.Y + within.Height - side), side);
    }

    private static bool Overlaps((int X, int Y, int Side) a, (int X, int Y, int Side) b) =>
        a.X < b.X + b.Side && b.X < a.X + a.Side && a.Y < b.Y + b.Side && b.Y < a.Y + a.Side;

    /// <summary>The positive's <paramref name="region"/> blended into <paramref name="twin"/> at the same place (moved inside the twin's content).</summary>
    private static bool Paste(Tensor pixels, int positive, int twin, (int X, int Y, int Side) region, LetterboxBox twinContent)
    {
        if (region.Side < 16 || region.Side > Math.Min(twinContent.Width, twinContent.Height))
            return false;
        var target = Square(region.X, region.Y, region.Side, twinContent);
        var patch = pixels[positive, .., TensorIndex.Slice(region.Y, region.Y + region.Side), TensorIndex.Slice(region.X, region.X + region.Side)];
        var into = pixels[twin, .., TensorIndex.Slice(target.Y, target.Y + target.Side), TensorIndex.Slice(target.X, target.X + target.Side)];
        var alpha = Ramp(region.Side, pixels.device).to_type(pixels.dtype);
        into.copy_(alpha * patch + (1 - alpha) * into);
        return true;
    }

    /// <summary>[side, side]: 1 inside, falling linearly to 0 over <see cref="Feather"/> pixels at the edges.</summary>
    public static Tensor Ramp(int side, Device device)
    {
        var steps = arange(side, device: device).to_type(ScalarType.Float32);
        var edge = minimum(steps + 1, side - steps).div(Feather).clamp_max(1f);
        return edge.unsqueeze(1) * edge.unsqueeze(0);
    }
}
