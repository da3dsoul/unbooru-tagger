using SkiaSharp;
using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

public class PerceptualHashTests
{
    private static string WriteImage(SKBitmap bitmap, SKEncodedImageFormat format, int quality)
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid():N}.img");
        using var data = bitmap.Encode(format, quality);
        using var stream = File.Create(path);
        data.SaveTo(stream);
        return path;
    }

    private static SKBitmap MakeGradientBitmap(int seed)
    {
        var bitmap = new SKBitmap(256, 256);
        using var canvas = new SKCanvas(bitmap);
        canvas.Clear(SKColors.White);

        using var gradientPaint = new SKPaint
        {
            Shader = SKShader.CreateLinearGradient(
                new SKPoint(0, 0),
                new SKPoint(256, 256),
                [new SKColor(20, 40, 200), new SKColor(220, 200, 30)],
                SKShaderTileMode.Clamp)
        };
        canvas.DrawRect(new SKRect(0, 0, 256, 256), gradientPaint);

        // A handful of solid shapes, offset by `seed`, give each variant distinct
        // large-scale structure the DCT's low frequencies will actually pick up —
        // a bare gradient alone is nearly featureless in the low-frequency band.
        using var shapePaint = new SKPaint { Color = new SKColor(200, 30, 30) };
        canvas.DrawCircle(64 + seed, 64, 40, shapePaint);
        canvas.DrawRect(new SKRect(140 + seed, 140, 220 + seed, 220), shapePaint);

        return bitmap;
    }

    [Fact]
    public void Compute_IsStable_AcrossReencodeAtDifferentQualityAndFormat()
    {
        using var bitmap = MakeGradientBitmap(seed: 0);
        var pngPath = WriteImage(bitmap, SKEncodedImageFormat.Png, 100);
        var jpegHighPath = WriteImage(bitmap, SKEncodedImageFormat.Jpeg, 95);
        var jpegLowPath = WriteImage(bitmap, SKEncodedImageFormat.Jpeg, 60);

        try
        {
            var pngHash = PerceptualHash.Compute(pngPath);
            var jpegHighHash = PerceptualHash.Compute(jpegHighPath);
            var jpegLowHash = PerceptualHash.Compute(jpegLowPath);

            Assert.True(PerceptualHash.HammingDistance(pngHash, jpegHighHash) <= 6);
            Assert.True(PerceptualHash.HammingDistance(pngHash, jpegLowHash) <= 6);
        }
        finally
        {
            File.Delete(pngPath);
            File.Delete(jpegHighPath);
            File.Delete(jpegLowPath);
        }
    }

    [Fact]
    public void Compute_DiffersSubstantially_ForVisuallyDifferentImages()
    {
        using var bitmapA = MakeGradientBitmap(seed: 0);
        using var bitmapB = MakeGradientBitmap(seed: 128);
        var pathA = WriteImage(bitmapA, SKEncodedImageFormat.Png, 100);
        var pathB = WriteImage(bitmapB, SKEncodedImageFormat.Png, 100);

        try
        {
            var hashA = PerceptualHash.Compute(pathA);
            var hashB = PerceptualHash.Compute(pathB);

            Assert.True(PerceptualHash.HammingDistance(hashA, hashB) > 6);
        }
        finally
        {
            File.Delete(pathA);
            File.Delete(pathB);
        }
    }

    [Fact]
    public void HammingDistance_ZeroForIdenticalHashes()
    {
        Assert.Equal(0, PerceptualHash.HammingDistance(0xABCDEF01UL, 0xABCDEF01UL));
    }

    [Fact]
    public void HammingDistance_CountsDifferingBits()
    {
        Assert.Equal(2, PerceptualHash.HammingDistance(0b1010UL, 0b0000UL));
    }
}
