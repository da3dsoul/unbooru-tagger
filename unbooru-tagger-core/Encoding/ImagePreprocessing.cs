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
        // Rgba8888/Unpremul so the raw byte layout is known (R,G,B,A per pixel)
        // regardless of platform-native decode format, and pixel values match what
        // GetPixel used to hand back (SKColor is always unpremultiplied) — needed to
        // read the buffer directly below instead of going through GetPixel per pixel.
        var resizeInfo = new SKImageInfo(inputSize, inputSize, SKColorType.Rgba8888, SKAlphaType.Unpremul);
        using var resized = original.Resize(resizeInfo, SKSamplingOptions.Default)
            ?? throw new InvalidOperationException("Failed to resize image.");

        var channelSize = inputSize * inputSize;
        var flat = new float[3 * channelSize];

        // Direct pixel-buffer access instead of GetPixel(x, y) per pixel: GetPixel's
        // per-call overhead (bounds checks, color conversion) dwarfs decode+resize cost
        // for a 224x224+ image, and this is the hottest loop in the whole preprocessing
        // pipeline since it runs once per image in the entire corpus.
        var pixels = resized.GetPixelSpan();
        var rowBytes = resized.RowBytes;
        const int bytesPerPixel = 4;

        for (var y = 0; y < inputSize; y++)
        {
            var rowStart = y * rowBytes;
            for (var x = 0; x < inputSize; x++)
            {
                var offset = rowStart + x * bytesPerPixel;
                var pixelIndex = y * inputSize + x;
                flat[(0 * channelSize) + pixelIndex] = (pixels[offset] / 255f - Mean[0]) / Std[0];
                flat[(1 * channelSize) + pixelIndex] = (pixels[offset + 1] / 255f - Mean[1]) / Std[1];
                flat[(2 * channelSize) + pixelIndex] = (pixels[offset + 2] / 255f - Mean[2]) / Std[2];
            }
        }

        return flat;
    }
}
