using SkiaSharp;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Runtime;
using UnbooruTagger.Core.Scoring;

namespace UnbooruTagger.Inference.Commands;

public static class HeatmapCommandHandler
{
    public static int Run(string modelDir, string imagePath, string tag, string outputPath)
    {
        using var model = ModelBundle.Load(modelDir);

        if (!model.Vocabulary.TryGet(tag, out var record))
        {
            Console.Error.WriteLine($"Unknown tag '{tag}'.");
            return 1;
        }

        var encoding = model.ImageEncoder.Encode(imagePath);
        var heatmap = TagScorer.Heatmap(model.Embeddings.GetRow(record.RowIndex), encoding.SpatialFeatures);

        using var original = SKBitmap.Decode(imagePath)
            ?? throw new InvalidDataException($"Could not decode image at '{imagePath}'.");
        using var overlay = RenderOverlay(original, heatmap, encoding.Content, model.ImageEncoder.InputSize);

        using var image = SKImage.FromBitmap(overlay);
        using var data = image.Encode(SKEncodedImageFormat.Png, quality: 100);
        using var stream = File.Create(outputPath);
        data.SaveTo(stream);

        return 0;
    }

    /// <summary>
    /// Upsamples the (typically much smaller than the image) heatmap grid, crops out the
    /// letterbox padding bars it covers alongside the real image content, and blends the
    /// result as a red overlay. <paramref name="content"/>/<paramref name="canvasSize"/>
    /// describe where the real image sits within the grid's full padded canvas — see
    /// <see cref="UnbooruTagger.Core.Encoding.ImageEncoding.Content"/>.
    /// </summary>
    private static SKBitmap RenderOverlay(SKBitmap original, float[,] heatmap, LetterboxBox content, int canvasSize)
    {
        var height = heatmap.GetLength(0);
        var width = heatmap.GetLength(1);

        var min = float.MaxValue;
        var max = float.MinValue;
        foreach (var value in heatmap)
        {
            min = Math.Min(min, value);
            max = Math.Max(max, value);
        }
        var range = Math.Max(max - min, 1e-6f);

        using var heatBitmap = new SKBitmap(width, height);
        for (var y = 0; y < height; y++)
        for (var x = 0; x < width; x++)
        {
            var normalized = (heatmap[y, x] - min) / range;
            heatBitmap.SetPixel(x, y, new SKColor(255, 0, 0, (byte)(normalized * 255)));
        }

        using var canvasHeat = heatBitmap.Resize(new SKImageInfo(canvasSize, canvasSize), SKSamplingOptions.Default)
            ?? throw new InvalidOperationException("Failed to resize heatmap overlay to canvas size.");

        using var croppedHeat = new SKBitmap(content.Width, content.Height);
        if (!canvasHeat.ExtractSubset(croppedHeat, new SKRectI(content.X, content.Y, content.X + content.Width, content.Y + content.Height)))
            throw new InvalidOperationException("Failed to crop heatmap overlay to the image content region.");

        using var resizedHeat = croppedHeat.Resize(new SKImageInfo(original.Width, original.Height), SKSamplingOptions.Default)
            ?? throw new InvalidOperationException("Failed to resize heatmap overlay to the original image size.");

        var result = new SKBitmap(original.Width, original.Height);
        using var canvas = new SKCanvas(result);
        canvas.DrawBitmap(original, 0, 0, SKSamplingOptions.Default, paint: null);
        canvas.DrawBitmap(resizedHeat, 0, 0, SKSamplingOptions.Default, paint: null);
        return result;
    }
}
