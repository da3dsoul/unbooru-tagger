using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// What a blanked rectangle is filled with, in the occlusion check and in erased-view training. Trained only on
/// zero-filled blanks, the hinge arm learned that a grey patch lowers a tag's score, so the check can be run with a fill
/// a model never trained on. No blur or pixelation (censoring is a tag of its own) and no patches of other images (they
/// can carry the very object).
/// </summary>
public enum OcclusionFill
{
    /// <summary>0, the normalized padding value.</summary>
    Zero,

    /// <summary>The image's mean color.</summary>
    Mean,

    /// <summary>Gaussian noise with the image's per-channel mean and spread.</summary>
    Noise,
}

public static class OcclusionFills
{
    /// <summary>The fills erased-view training draws from (one per pair, the same for its object and control view).</summary>
    public static readonly OcclusionFill[] Training = [OcclusionFill.Mean, OcclusionFill.Noise];

    public static OcclusionFill Parse(string name) => Enum.Parse<OcclusionFill>(name, ignoreCase: true);

    /// <summary>
    /// Fills <paramref name="rectangles"/> (canvas [x, y, w, h]) of a [3, size, size] image in place.
    /// </summary>
    public static void Apply(Span<float> image, IEnumerable<int[]> rectangles, int size, OcclusionFill fill, Random random)
    {
        var channel = size * size;
        Span<float> mean = stackalloc float[3];
        Span<float> spread = stackalloc float[3];
        for (var ch = 0; ch < 3; ch++)
        {
            double sum = 0, squares = 0;
            foreach (var v in image.Slice(ch * channel, channel))
                (sum, squares) = (sum + v, squares + v * v);
            mean[ch] = (float)(sum / channel);
            spread[ch] = (float)Math.Sqrt(Math.Max(0, squares / channel - mean[ch] * mean[ch]));
        }

        foreach (var r in rectangles)
            for (var ch = 0; ch < 3; ch++)
                for (var y = Math.Max(0, r[1]); y < Math.Min(size, r[1] + r[3]); y++)
                    for (var x = Math.Max(0, r[0]); x < Math.Min(size, r[0] + r[2]); x++)
                    {
                        var index = ch * channel + y * size + x;
                        image[index] = fill switch
                        {
                            OcclusionFill.Zero => 0,
                            OcclusionFill.Mean => mean[ch],
                            OcclusionFill.Noise => mean[ch] + spread[ch] * Gaussian(random),
                            _ => throw new ArgumentOutOfRangeException(nameof(fill)),
                        };
                    }
    }

    /// <summary>
    /// [n, 3, size, size] fill images for <paramref name="pixels"/> ([n, 3, size, size]), image k filled with
    /// <paramref name="fills"/>[k].
    /// </summary>
    public static Tensor Images(Tensor pixels, IReadOnlyList<OcclusionFill> fills)
    {
        var mean = pixels.mean([2, 3], keepdim: true);
        var spread = pixels.std([2, 3], keepdim: true);
        var n = pixels.shape[0];
        var choices = new List<Tensor>();
        for (var k = 0; k < n; k++)
        {
            var image = pixels[k].unsqueeze(0);
            choices.Add(fills[k] switch
            {
                OcclusionFill.Zero => zeros_like(image),
                OcclusionFill.Mean => mean[k].unsqueeze(0).expand_as(image),
                OcclusionFill.Noise => mean[k].unsqueeze(0) + spread[k].unsqueeze(0) * randn_like(image),
                _ => throw new ArgumentOutOfRangeException(nameof(fills)),
            });
        }

        return cat(choices, 0);
    }

    private static float Gaussian(Random random) =>
        (float)(Math.Sqrt(-2 * Math.Log(1 - random.NextDouble())) * Math.Cos(2 * Math.PI * random.NextDouble()));
}
