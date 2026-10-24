using System.Text.Json;
using SkiaSharp;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Runtime;
using UnbooruTagger.Core.Scoring;

namespace UnbooruTagger.Inference.Commands;

public static class DetectCommandHandler
{
    // Picked by a stable hash of the tag name, so a tag gets the same color in every render --
    // which is what lets a gold render and a model render of one image be compared by eye.
    private static readonly SKColor[] Palette =
    [
        SKColors.Red, SKColors.DodgerBlue, SKColors.Lime, SKColors.Gold,
        SKColors.Magenta, SKColors.Cyan, SKColors.Orange, SKColors.MediumPurple
    ];

    // Resolution the guide image and every tag's heatmap are refined at (see HeatmapRefiner) —
    // capped independent of the original image's size so refinement cost stays bounded.
    private const int RefinementSize = 160;

    public static int Run(
        string modelDir, string imagePath, float? threshold, float boxThreshold, float boxPercentile, float minHeatmapRange,
        float minBoxFraction, int maxBoxesPerTag, string? outputPath)
    {
        using var model = ModelBundle.Load(modelDir);

        var detections = Detect(model, imagePath, threshold, boxThreshold, boxPercentile, minHeatmapRange, minBoxFraction, maxBoxesPerTag);

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
    /// Scores every vocabulary tag against the whole image, keeps each tag above its
    /// threshold (<paramref name="threshold"/>, or the tag's calibrated one when null — see
    /// <see cref="ModelBundle.ThresholdFor"/>), drops the weaker side of any contradictory
    /// pair (<see cref="ModelBundle.ExclusiveGroups"/>), then for each survivor: builds its heatmap, refines it against the image with
    /// <see cref="HeatmapRefiner.Refine"/> so the boxes snap to real edges instead of the raw
    /// grid's blocky boundary, and thresholds it into boxes via <see cref="TagScorer.DetectBoxes"/>
    /// using <paramref name="boxThreshold"/> as an absolute floor, <paramref name="boxPercentile"/>
    /// to additionally cut within that tag's own heatmap range, and <paramref name="minHeatmapRange"/>
    /// to suppress boxes entirely for a tag whose heatmap has no real spatial signal to cut into.
    /// Regions smaller than <paramref name="minBoxFraction"/> of the grid are dropped as specks,
    /// and at most <paramref name="maxBoxesPerTag"/> (strongest peak first; 0 = unlimited) are kept.
    /// Boxes are reported in the original image's pixel space, not the resized model-input space.
    /// </summary>
    public static List<TagDetection> Detect(
        ModelBundle model, string imagePath, float? threshold, float boxThreshold, float boxPercentile, float minHeatmapRange = 0.05f,
        float minBoxFraction = 0f, int maxBoxesPerTag = 0)
    {
        var encoding = model.ImageEncoder.Encode(imagePath);

        using var bitmap = SKBitmap.Decode(imagePath)
            ?? throw new InvalidDataException($"Could not decode image at '{imagePath}'.");

        var guide = HeatmapRefiner.BuildGuide(bitmap, RefinementSize, out var refinementContent);

        var candidates = new List<(int Row, float Confidence)>();
        for (var row = 0; row < model.Embeddings.RowCount; row++)
        {
            var confidence = TagScorer.Score(encoding.PooledEmbedding, model.Embeddings.GetRow(row), model.LogitScaling);
            if (confidence >= model.ThresholdFor(row, threshold))
                candidates.Add((row, confidence));
        }

        candidates.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));

        var detections = new List<TagDetection>();
        foreach (var (row, confidence) in model.ExclusiveGroups.Resolve(candidates, c => c.Row))
        {
            var tag = model.Vocabulary.GetByRowIndex(row).Tag;
            if (!model.Localizability.IsLocalizable(tag))
            {
                // Whole-image tag (background, style, framing, ...): reported, never boxed.
                detections.Add(new TagDetection(tag, confidence, []));
                continue;
            }

            var tagEmbedding = model.Embeddings.GetRow(row);
            var heatmap = TagScorer.Heatmap(tagEmbedding, encoding.SpatialFeatures, model.LogitScaling);
            var refinedHeatmap = HeatmapRefiner.Refine(heatmap, guide);
            var boxes = TagScorer.DetectBoxes(
                refinedHeatmap, boxThreshold, boxPercentile, refinementContent, RefinementSize, bitmap.Width, bitmap.Height, minHeatmapRange,
                minBoxFraction, maxBoxesPerTag);
            detections.Add(new TagDetection(model.Vocabulary.GetByRowIndex(row).Tag, confidence, boxes));
        }

        detections.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
        return detections;
    }

    /// <summary>Draws one outlined rectangle per box plus a "tag confidence" label above it (just "tag" when <paramref name="showConfidence"/> is false, e.g. for hand-drawn gold boxes), colored per tag.</summary>
    internal static SKBitmap RenderBoxes(SKBitmap original, IReadOnlyList<TagDetection> detections, bool showConfidence = true)
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
            // Gold renders mark group/ignore boxes with a " [group]" / " [unsure]" suffix: same
            // color as the plain tag, and unsure ones dashed.
            var suffixAt = detection.Tag.IndexOf(" [", StringComparison.Ordinal);
            var color = ColorFor(suffixAt >= 0 ? detection.Tag[..suffixAt] : detection.Tag);
            boxPaint.Color = color;
            labelPaint.Color = color;
            using var dash = detection.Tag.EndsWith("[unsure]", StringComparison.Ordinal) ? SKPathEffect.CreateDash([6, 4], 0) : null;
            boxPaint.PathEffect = dash;

            foreach (var box in detection.Boxes)
            {
                var rect = new SKRect(box.X, box.Y, box.X + box.Width, box.Y + box.Height);
                canvas.DrawRect(rect, boxPaint);

                // The box's own peak, not the whole-image confidence -- otherwise every box of
                // a tag carries the same number and a weak stray region reads as strong as the
                // real one.
                var label = showConfidence ? $"{detection.Tag} {box.Confidence:F2}" : detection.Tag;
                var textWidth = font.MeasureText(label, textPaint);
                var labelTop = Math.Max(rect.Top - 18, 0);
                var labelRect = new SKRect(rect.Left, labelTop, rect.Left + textWidth + 6, labelTop + 18);

                canvas.DrawRect(labelRect, labelPaint);
                canvas.DrawText(label, labelRect.Left + 3, labelRect.Bottom - 4, SKTextAlign.Left, font, textPaint);
            }
        }

        return result;
    }

    internal static SKColor ColorFor(string tag)
    {
        // FNV-1a: string.GetHashCode is randomized per process, which would reshuffle colors every run.
        var hash = 2166136261u;
        foreach (var c in tag)
            hash = (hash ^ c) * 16777619u;
        return Palette[hash % (uint)Palette.Length];
    }
}
