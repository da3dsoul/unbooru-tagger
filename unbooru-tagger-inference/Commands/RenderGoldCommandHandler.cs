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
///
/// With a model, the boxes also go to <see cref="BoxesFileName"/> (per image: each detected tag,
/// its confidence and its boxes, in image pixels), which gold/compare.html draws on a canvas over
/// the plain image with one list entry per box, so overlapping boxes can be told apart. The PNGs
/// are then only written with <c>--png</c>.
/// </summary>
public static class RenderGoldCommandHandler
{
    private const int HeaderHeight = 24;
    public const string BoxesFileName = "boxes.json";

    public static int Run(
        string goldPath, string outputDirectory, string? modelDir, float? threshold,
        float boxThreshold, float boxPercentile, float minHeatmapRange, float minBoxFraction, int maxBoxesPerTag, bool allTags = false, bool png = false)
    {
        var goldDirectory = Path.GetDirectoryName(Path.GetFullPath(goldPath))!;
        using var document = JsonDocument.Parse(File.ReadAllText(goldPath));
        using var model = modelDir is not null ? ModelBundle.Load(modelDir) : null;
        Directory.CreateDirectory(outputDirectory);

        var count = 0;
        var boxes = new Dictionary<string, object>();
        var writePng = model is null || png;
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
            using var goldRender = writePng ? DetectCommandHandler.RenderBoxes(original, gold, showConfidence: false) : null;

            SKBitmap? output = null;
            if (model is not null)
            {
                var detections = DetectCommandHandler.Detect(model, imagePath, threshold, boxThreshold, boxPercentile, minHeatmapRange, minBoxFraction, maxBoxesPerTag);
                if (!allTags)
                {
                    // Only the tags the gold set boxes on this image, so both panels answer the same question.
                    var targets = image.GetProperty("targets").EnumerateArray().Select(t => t.GetString()!).ToHashSet(StringComparer.Ordinal);
                    detections = detections.Where(d => targets.Contains(d.Tag) || targets.Contains(d.Tag.Replace(' ', '_'))).ToList();
                }
                boxes[index.ToString()] = detections.Select(d => new
                {
                    tag = d.Tag,
                    confidence = d.Confidence,
                    boxes = d.Boxes.Select(b => new[] { b.X, b.Y, b.Width, b.Height, b.Confidence }),
                });
                if (writePng)
                {
                    using var modelRender = DetectCommandHandler.RenderBoxes(original, detections);
                    output = SideBySide(($"gold ({StratumOf(image)})", goldRender!), ($"model ({ModelName(modelDir!)})", modelRender));
                }
            }
            else
            {
                output = SideBySide(($"gold ({StratumOf(image)})", goldRender!));
            }

            if (output is not null)
            {
                using (output)
                using (var encoded = SKImage.FromBitmap(output).Encode(SKEncodedImageFormat.Png, 100))
                using (var stream = File.Create(Path.Combine(outputDirectory, $"{index}.png")))
                    encoded.SaveTo(stream);
            }
            count++;
        }

        if (model is not null)
            File.WriteAllText(Path.Combine(outputDirectory, BoxesFileName), JsonSerializer.Serialize(new { model = ModelName(modelDir!), images = boxes }));
        Console.WriteLine(writePng
            ? $"Rendered {count} images to '{outputDirectory}'."
            : $"Wrote {count} images' boxes to '{Path.Combine(outputDirectory, BoxesFileName)}'.");
        return 0;
    }

    private static string ModelName(string modelDir) => Path.GetFileName(Path.TrimEndingDirectorySeparator(modelDir));

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
