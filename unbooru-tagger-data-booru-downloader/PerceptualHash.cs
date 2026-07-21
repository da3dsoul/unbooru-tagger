using SkiaSharp;

namespace UnbooruTagger.Crawler;

/// <summary>
/// 64-bit DCT-based perceptual hash (the standard "pHash" algorithm: grayscale, resize
/// to 32x32, 2D DCT, threshold the low-frequency 8x8 block against its own median).
/// MD5 alone misses cross-site duplicates: Danbooru and Gelbooru routinely re-encode or
/// re-compress the same source image before serving it, which changes every byte (and
/// thus the md5) without changing what the image looks like. Unlike a plain average-hash,
/// the DCT's energy compaction concentrates on real image structure rather than exact
/// pixel values, so it survives re-encoding/resizing while still separating genuinely
/// different images.
/// </summary>
public static class PerceptualHash
{
    private const int SampleSize = 32;
    private const int HashSize = 8;

    public static ulong Compute(string imagePath)
    {
        using var bitmap = SKBitmap.Decode(imagePath)
            ?? throw new InvalidDataException($"Could not decode image at '{imagePath}'.");
        return Compute(bitmap);
    }

    public static ulong Compute(SKBitmap bitmap)
    {
        var samples = ToGrayscaleSamples(bitmap, SampleSize);
        var dct = Dct2D(samples, SampleSize);

        var lowFreq = new double[HashSize * HashSize];
        for (var y = 0; y < HashSize; y++)
            for (var x = 0; x < HashSize; x++)
                lowFreq[y * HashSize + x] = dct[y * SampleSize + x];

        var sorted = (double[])lowFreq.Clone();
        Array.Sort(sorted);
        var median = (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2.0;

        var hash = 0UL;
        foreach (var value in lowFreq)
        {
            hash <<= 1;
            if (value > median)
                hash |= 1;
        }
        return hash;
    }

    /// <summary>Bit-differing count between two hashes — 0 is identical, 64 is maximally different.</summary>
    public static int HammingDistance(ulong a, ulong b) => System.Numerics.BitOperations.PopCount(a ^ b);

    private static double[] ToGrayscaleSamples(SKBitmap original, int size)
    {
        var resizeInfo = new SKImageInfo(size, size, SKColorType.Rgba8888, SKAlphaType.Unpremul);
        using var resized = original.Resize(resizeInfo, SKSamplingOptions.Default)
            ?? throw new InvalidOperationException("Failed to resize image for perceptual hashing.");

        var samples = new double[size * size];
        var pixels = resized.GetPixelSpan();
        var rowBytes = resized.RowBytes;
        const int bytesPerPixel = 4;

        for (var y = 0; y < size; y++)
        {
            var rowStart = y * rowBytes;
            for (var x = 0; x < size; x++)
            {
                var offset = rowStart + x * bytesPerPixel;
                // Rec. 601 luma weights.
                samples[y * size + x] = (0.299 * pixels[offset]) + (0.587 * pixels[offset + 1]) + (0.114 * pixels[offset + 2]);
            }
        }
        return samples;
    }

    /// <summary>Separable 2D DCT-II (rows then columns) — O(size^3), trivial at 32x32.</summary>
    private static double[] Dct2D(double[] samples, int size)
    {
        var afterRows = new double[size * size];
        for (var y = 0; y < size; y++)
        {
            for (var u = 0; u < size; u++)
            {
                var sum = 0.0;
                for (var x = 0; x < size; x++)
                    sum += samples[(y * size) + x] * Math.Cos(Math.PI / size * (x + 0.5) * u);
                afterRows[(y * size) + u] = sum;
            }
        }

        var result = new double[size * size];
        for (var u = 0; u < size; u++)
        {
            for (var v = 0; v < size; v++)
            {
                var sum = 0.0;
                for (var y = 0; y < size; y++)
                    sum += afterRows[(y * size) + u] * Math.Cos(Math.PI / size * (y + 0.5) * v);
                result[(v * size) + u] = sum;
            }
        }
        return result;
    }
}
