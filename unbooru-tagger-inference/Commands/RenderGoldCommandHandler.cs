using System.Text.Json;
using SkiaSharp;
using UnbooruTagger.Core.Runtime;
using UnbooruTagger.Core.Scoring;

namespace UnbooruTagger.Inference.Commands;

/// <summary>
/// Renders a gold set's hand-checked boxes the same way <c>detect</c> renders model output
/// (same drawing code, same per-tag colors), one PNG per image. With a model, each PNG puts the
/// gold boxes and that model's <c>detect</c> output side by side, so localization can be judged
/// by eye against ground truth image by image. The model panel shows only the tags the gold set
/// boxes on that image unless <c>--all-tags</c> is given.
/// </summary>
public static class RenderGoldCommandHandler
{
    private const int HeaderHeight = 24;

    public static int Run(
        string goldPath, string outputDirectory, string? modelDir, float? threshold,
        float boxThreshold, float boxPercentile, float minHeatmapRange, float minBoxFraction, int maxBoxesPerTag, bool allTags = false)
    {
        var goldDirectory = Path.GetDirectoryName(Path.GetFullPath(goldPath))!;
        using var document = JsonDocument.Parse(File.ReadAllText(goldPath));
        using var model = modelDir is not null ? ModelBundle.Load(modelDir) : null;
        Directory.CreateDirectory(outputDirectory);

        var count = 0;
        foreach (var image in document.RootElement.GetProperty("images").EnumerateArray())
        {
            if (image.TryGetProperty("excluded", out var excluded) && excluded.ValueKind == JsonValueKind.True)
                continue;

            var index = image.GetProperty("cacheIndex").GetInt32();
            var imagePath = Path.Combine(goldDirectory, image.GetProperty("file").GetString()!);
            using var original = SKBitmap.Decode(imagePath)
                ?? throw new InvalidDataException($"Could not decode '{imagePath}'.");

            var gold = image.GetProperty("instances").EnumerateArray()
                .Select(i =>
                {
                    var kind = i.TryGetProperty("kind", out var k) ? k.GetString() : null;
                    var tag = i.GetProperty("tag").GetString()!;
                    return (Tag: kind switch { "group" => $"{tag} [group]", "ignore" => $"{tag} [unsure]", _ => tag },
                        Box: i.GetProperty("box").EnumerateArray().Select(v => (int)Math.Round(v.GetDouble())).ToArray());
                })
                .GroupBy(i => i.Tag)
                .Select(g => new TagDetection(g.Key, 1f, g.Select(i => new BoundingBox(i.Box[0], i.Box[1], i.Box[2], i.Box[3], 1f)).ToList()))
                .ToList();
            using var goldRender = DetectCommandHandler.RenderBoxes(original, gold, showConfidence: false);

            SKBitmap output;
            if (model is not null)
            {
                var detections = DetectCommandHandler.Detect(model, imagePath, threshold, boxThreshold, boxPercentile, minHeatmapRange, minBoxFraction, maxBoxesPerTag);
                if (!allTags)
                {
                    // Only the tags the gold set boxes on this image, so both panels answer the same question.
                    var targets = image.GetProperty("targets").EnumerateArray().Select(t => t.GetString()!).ToHashSet(StringComparer.Ordinal);
                    detections = detections.Where(d => targets.Contains(d.Tag) || targets.Contains(d.Tag.Replace(' ', '_'))).ToList();
                }
                using var modelRender = DetectCommandHandler.RenderBoxes(original, detections);
                output = SideBySide(($"gold ({StratumOf(image)})", goldRender), ($"model ({Path.GetFileName(Path.TrimEndingDirectorySeparator(modelDir!))})", modelRender));
            }
            else
            {
                output = SideBySide(($"gold ({StratumOf(image)})", goldRender));
            }

            using (output)
            using (var encoded = SKImage.FromBitmap(output).Encode(SKEncodedImageFormat.Png, 100))
            using (var stream = File.Create(Path.Combine(outputDirectory, $"{index}.png")))
                encoded.SaveTo(stream);
            count++;
        }

        Console.WriteLine($"Rendered {count} images to '{outputDirectory}'.");
        return 0;
    }

    private static string StratumOf(JsonElement image) =>
        image.TryGetProperty("stratum", out var stratum) ? stratum.GetString() ?? "clean" : "clean";

    private static SKBitmap SideBySide(params (string Title, SKBitmap Image)[] panels)
    {
        const int gap = 8;
        var width = panels.Sum(p => p.Image.Width) + gap * (panels.Length - 1);
        var height = panels.Max(p => p.Image.Height) + HeaderHeight;
        var result = new SKBitmap(width, height);
        using var canvas = new SKCanvas(result);
        canvas.Clear(SKColors.White);
        using var font = new SKFont(SKTypeface.Default, size: 16);
        using var textPaint = new SKPaint { Color = SKColors.Black, IsAntialias = true };

        var x = 0;
        foreach (var (title, image) in panels)
        {
            canvas.DrawText(title, x + 4, HeaderHeight - 6, SKTextAlign.Left, font, textPaint);
            canvas.DrawBitmap(image, x, HeaderHeight, SKSamplingOptions.Default, paint: null);
            x += image.Width + gap;
        }

        return result;
    }
}
