namespace UnbooruTagger.Core.Encoding;

/// <summary>
/// The pre-pool spatial feature map from the image tower (height x width x channels,
/// row-major). This is what makes rough per-tag localization possible without a
/// separate CAM/Grad-CAM step (see CLAUDE.md's localization section).
/// </summary>
public sealed class SpatialFeatureMap
{
    private readonly float[] _data;

    public int Height { get; }
    public int Width { get; }
    public int Channels { get; }

    public SpatialFeatureMap(float[] data, int height, int width, int channels)
    {
        if (data.Length != height * width * channels)
            throw new ArgumentException("Data length does not match height * width * channels.");

        _data = data;
        Height = height;
        Width = width;
        Channels = channels;
    }

    /// <summary>The channel vector at spatial location (y, x).</summary>
    public ReadOnlySpan<float> this[int y, int x] =>
        _data.AsSpan((y * Width + x) * Channels, Channels);
}

/// <summary>
/// The image tower's output: a pooled global embedding plus the spatial map it was
/// pooled from, plus where the source image's real content landed inside the
/// (letterboxed) input canvas — <see cref="SpatialFeatures"/> covers the whole canvas,
/// padding included, so callers mapping grid cells back to the original image's pixel
/// space need <see cref="Content"/> to exclude the padding bars.
/// </summary>
public sealed record ImageEncoding(float[] PooledEmbedding, SpatialFeatureMap SpatialFeatures, LetterboxBox Content);
