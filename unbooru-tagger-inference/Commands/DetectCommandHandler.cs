using System.Text.Json;
using SkiaSharp;
using UnbooruTagger.Core.Runtime;
using UnbooruTagger.Core.Scoring;

namespace UnbooruTagger.Inference.Commands;

public static class DetectCommandHandler
{
    // Cycled by detection index so distinct tags read as distinct colors in the rendered output.
    private static readonly SKColor[] Palette =
    [
        SKColors.Red, SKColors.DodgerBlue, SKColors.Lime, SKColors.Gold,
        SKColors.Magenta, SKColors.Cyan, SKColors.Orange, SKColors.MediumPurple
    ];

    // Resolution the guide image and every tag's heatmap are refined at (see HeatmapRefiner) —
    // capped independent of the original image's size so refinement cost stays bounded.
    private const int RefinementSize = 160;

    public static int Run(string modelDir, string imagePath, float threshold, float boxThreshold, float boxPercentile, string? outputPath)
    {
        using var model = ModelBundle.Load(modelDir);

        var detections = Detect(model, imagePath, threshold, boxThreshold, boxPercentile);

        Console.WriteLine(JsonSerializer.Serialize(detections, new JsonSerializerOptions { WriteIndented = true }));

        if (outputPath is not null)
        {
            using var original = SKBitmap.Decode(imagePath)
                ?? throw new InvalidDataException($"Could not decode image at '{imagePath}'.");
            using var annotated = RenderBoxes(original, detections);

            using var image = SKImage.FromBitmap(annotated);
            using var data = image.Encode(SKEncodedImageFormat.Png, quality: 100);
            using var stream = File.Create(outputPath);
            data.SaveTo(stream);
        }

        return 0;
    }

    /// <summary>
    /// Scores every vocabulary tag against the whole image, then, for each tag above
    /// <paramref name="threshold"/>: builds its heatmap, refines it against the image with
    /// <see cref="HeatmapRefiner.Refine"/> so the boxes snap to real edges instead of the raw
    /// grid's blocky boundary, and thresholds it into boxes via <see cref="TagScorer.DetectBoxes"/>
    /// using <paramref name="boxThreshold"/> as an absolute floor and <paramref name="boxPercentile"/>
    /// to additionally cut within that tag's own heatmap range. Boxes are reported in the
    /// original image's pixel space, not the resized model-input space.
    /// </summary>
    public static List<TagDetection> Detect(
        ModelBundle model, string imagePath, float threshold, float boxThreshold, float boxPercentile)
    {
        var encoding = model.ImageEncoder.Encode(imagePath);

        using var bitmap = SKBitmap.Decode(imagePath)
            ?? throw new InvalidDataException($"Could not decode image at '{imagePath}'.");

        var guide = HeatmapRefiner.BuildGuide(bitmap, RefinementSize);

        var detections = new List<TagDetection>();
        for (var row = 0; row < model.Embeddings.RowCount; row++)
        {
            var tagEmbedding = model.Embeddings.GetRow(row);
            var confidence = TagScorer.Score(encoding.PooledEmbedding, tagEmbedding);
            if (confidence < threshold)
                continue;

            var heatmap = TagScorer.Heatmap(tagEmbedding, encoding.SpatialFeatures);
            var refinedHeatmap = HeatmapRefiner.Refine(heatmap, guide);
            var boxes = TagScorer.DetectBoxes(refinedHeatmap, boxThreshold, boxPercentile, bitmap.Width, bitmap.Height);
            detections.Add(new TagDetection(model.Vocabulary.GetByRowIndex(row).Tag, confidence, boxes));
        }

        detections.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
        return detections;
    }

    /// <summary>Draws one outlined rectangle per box plus a "tag confidence" label above it, cycling a color per tag so overlapping detections stay distinguishable.</summary>
    private static SKBitmap RenderBoxes(SKBitmap original, IReadOnlyList<TagDetection> detections)
    {
        var result = new SKBitmap(original.Width, original.Height);
        using var canvas = new SKCanvas(result);
        canvas.DrawBitmap(original, 0, 0, SKSamplingOptions.Default, paint: null);

        using var font = new SKFont(SKTypeface.Default, size: 14);
        using var boxPaint = new SKPaint { Style = SKPaintStyle.Stroke, StrokeWidth = 2, IsAntialias = true };
        using var labelPaint = new SKPaint { Style = SKPaintStyle.Fill };
        using var textPaint = new SKPaint { Color = SKColors.Black, IsAntialias = true };

        for (var i = 0; i < detections.Count; i++)
        {
            var detection = detections[i];
            var color = Palette[i % Palette.Length];
            boxPaint.Color = color;
            labelPaint.Color = color;

            foreach (var box in detection.Boxes)
            {
                var rect = new SKRect(box.X, box.Y, box.X + box.Width, box.Y + box.Height);
                canvas.DrawRect(rect, boxPaint);

                var label = $"{detection.Tag} {detection.Confidence:F2}";
                var textWidth = font.MeasureText(label, textPaint);
                var labelTop = Math.Max(rect.Top - 18, 0);
                var labelRect = new SKRect(rect.Left, labelTop, rect.Left + textWidth + 6, labelTop + 18);

                canvas.DrawRect(labelRect, labelPaint);
                canvas.DrawText(label, labelRect.Left + 3, labelRect.Bottom - 4, SKTextAlign.Left, font, textPaint);
            }
        }

        return result;
    }
}
