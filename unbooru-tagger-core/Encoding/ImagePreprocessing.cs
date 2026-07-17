using SkiaSharp;

namespace UnbooruTagger.Core.Encoding;

/// <summary>
/// The single source of truth for image decode/resize/normalize, shared by
/// <see cref="OnnxImageEncoder"/> (inference), Training's per-epoch batch loader, and
/// the data pipeline's bulk preprocessor — they must all agree on this or a trained
/// model's normalization will silently mismatch what it sees at inference time.
/// </summary>
public static class ImagePreprocessing
{
    public static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    public static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    /// <summary>Decodes, resizes, and normalizes one image into a flat NCHW-ordered (channels-first) <c>float[3 * inputSize * inputSize]</c>.</summary>
    public static float[] LoadAndNormalize(string imagePath, int inputSize)
    {
        using var original = SKBitmap.Decode(imagePath)
            ?? throw new InvalidDataException($"Could not decode image at '{imagePath}'.");
        return Normalize(original, inputSize);
    }

    /// <summary>Same as <see cref="LoadAndNormalize(string, int)"/> but decodes from an in-memory stream (e.g. a DB blob) instead of a file path.</summary>
    public static float[] LoadAndNormalize(Stream imageStream, int inputSize)
    {
        using var original = SKBitmap.Decode(imageStream)
            ?? throw new InvalidDataException("Could not decode image from the given stream.");
        return Normalize(original, inputSize);
    }

    private static float[] Normalize(SKBitmap original, int inputSize)
    {
        using var resized = original.Resize(new SKImageInfo(inputSize, inputSize), SKSamplingOptions.Default)
            ?? throw new InvalidOperationException("Failed to resize image.");

        var channelSize = inputSize * inputSize;
        var flat = new float[3 * channelSize];

        for (var y = 0; y < inputSize; y++)
        for (var x = 0; x < inputSize; x++)
        {
            var pixel = resized.GetPixel(x, y);
            var pixelIndex = y * inputSize + x;
            flat[(0 * channelSize) + pixelIndex] = (pixel.Red / 255f - Mean[0]) / Std[0];
            flat[(1 * channelSize) + pixelIndex] = (pixel.Green / 255f - Mean[1]) / Std[1];
            flat[(2 * channelSize) + pixelIndex] = (pixel.Blue / 255f - Mean[2]) / Std[2];
        }

        return flat;
    }
}
