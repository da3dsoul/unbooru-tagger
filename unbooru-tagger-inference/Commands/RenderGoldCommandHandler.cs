using System.Text.Json;
using SkiaSharp;
using UnbooruTagger.Core.Encoding;
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
///
/// <c>--scores</c> takes the gold pairs' scores from <c>evaluate --dump-occlusion</c>'s occlusion_pairs.jsonl (its
/// unblanked "Full" score) instead of the bundle, for a checkpoint the bundle can't score (glimpses, exported with
/// <c>export-onnx --ignore-pooling</c>): the bundle still draws the boxes. Only those pairs' tags are drawn, even with <c>--all-tags</c>.
///
/// Each tag in <see cref="BoxesFileName"/> carries its calibrated threshold. <c>--min-confidence</c> keeps the gold set's tags
/// down to that confidence instead of their thresholds, so compare.html's cutoff slider can lower the cutoff live.
/// </summary>
public static class RenderGoldCommandHandler
{
    private const int HeaderHeight = 24;
    public const string BoxesFileName = "boxes.json";

    public static int Run(
        string goldPath, string outputDirectory, string? modelDir, float? threshold,
        float boxThreshold, float boxPercentile, float minHeatmapRange, float minBoxFraction, int maxBoxesPerTag, bool allTags = false, bool png = false,
        string? scoresPath = null, float? minConfidence = null, InferenceDevice device = InferenceDevice.Cpu, int batchSize = 16)
    {
        var goldDirectory = Path.GetDirectoryName(Path.GetFullPath(goldPath))!;
        using var document = JsonDocument.Parse(File.ReadAllText(goldPath));
        using var model = modelDir is not null ? ModelBundle.Load(modelDir, device: device) : null;
        var encodeTime = new System.Diagnostics.Stopwatch();
        var boxTime = new System.Diagnostics.Stopwatch();
        Directory.CreateDirectory(outputDirectory);
        var scores = scoresPath is not null ? LoadScores(scoresPath) : null;

        var count = 0;
        var boxes = new Dictionary<string, object>();
        var writePng = model is null || png;
        var images = document.RootElement.GetProperty("images").EnumerateArray()
            .Where(i => !(i.TryGetProperty("excluded", out var excluded) && excluded.ValueKind == JsonValueKind.True))
            .ToList();
        // The model runs batchSize images at a time; everything after it is per image.
        foreach (var chunk in images.Chunk(Math.Max(1, batchSize)))
        {
            var bitmaps = new List<SKBitmap>();
            try
            {
                foreach (var image in chunk)
                {
                    var path = Path.Combine(goldDirectory, image.GetProperty("file").GetString()!);
                    bitmaps.Add(SKBitmap.Decode(path) ?? throw new InvalidDataException($"Could not decode '{path}'."));
                }
                IReadOnlyList<ImageEncoding>? encodings = null;
                if (model is not null)
                {
                    encodeTime.Start();
                    encodings = model.ImageEncoder.EncodeBatch(bitmaps.Select(b => ImagePreprocessing.Normalize(b, model.ImageEncoder.InputSize)).ToList());
                    encodeTime.Stop();
                }
                // Heatmaps and boxes are CPU work per image, so the batch's images run in parallel.
                var detected = new List<TagDetection>[chunk.Length];
                if (model is not null)
                {
                    boxTime.Start();
                    Parallel.For(0, chunk.Length, c =>
                    {
                        // Only the tags the gold set boxes on this image, so both panels answer the same question. Scored only for
                        // those: down to a low floor, every vocabulary row would cost a heatmap.
                        var targets = allTags ? null
                            : chunk[c].GetProperty("targets").EnumerateArray().Select(t => t.GetString()!.Replace(' ', '_')).ToHashSet(StringComparer.Ordinal);
                        detected[c] = DetectCommandHandler.Detect(
                            model, encodings![c], DetectCommandHandler.RefinementGuide.From(bitmaps[c]), minConfidence ?? threshold, boxThreshold,
                            boxPercentile, minHeatmapRange, minBoxFraction, maxBoxesPerTag, scores?.GetValueOrDefault(chunk[c].GetProperty("cacheIndex").GetInt32()),
                            targets);
                    });
                    boxTime.Stop();
                }

                for (var c = 0; c < chunk.Length; c++)
                {
                    var image = chunk[c];
                    var index = image.GetProperty("cacheIndex").GetInt32();
                    var original = bitmaps[c];

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
                        var detections = detected[c];
                        boxes[index.ToString()] = detections.Select(d => new
                        {
                            tag = d.Tag,
                            confidence = d.Confidence,
                            threshold = model.Vocabulary.TryGet(d.Tag, out var record) ? model.ThresholdFor(record.RowIndex, threshold) : (float?)null,
                            boxes = d.Boxes.Select(b => new[] { b.X, b.Y, b.Width, b.Height, b.Confidence }),
                        });
                        if (writePng)
                        {
                            // A PNG can't move its cutoff, so it keeps the tags' thresholds.
                            using var modelRender = DetectCommandHandler.RenderBoxes(original, minConfidence is null ? detections
                                : detections.Where(d => !model.Vocabulary.TryGet(d.Tag, out var r) || d.Confidence >= model.ThresholdFor(r.RowIndex, threshold)).ToList());
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
            }
            finally
            {
                foreach (var bitmap in bitmaps)
                    bitmap.Dispose();
            }
        }

        if (model is not null)
            File.WriteAllText(Path.Combine(outputDirectory, BoxesFileName), JsonSerializer.Serialize(new { model = ModelName(modelDir!), images = boxes }));
        if (model is not null)
            Console.WriteLine($"Model {encodeTime.Elapsed.TotalSeconds:F1}s, heatmaps and boxes {boxTime.Elapsed.TotalSeconds:F1}s, over {count} images.");
        Console.WriteLine(writePng
            ? $"Rendered {count} images to '{outputDirectory}'."
            : $"Wrote {count} images' boxes to '{Path.Combine(outputDirectory, BoxesFileName)}'.");
        return 0;
    }

    /// <summary>Per gold image (cache index), each pair's unblanked score from an occlusion_pairs.jsonl.</summary>
    private static Dictionary<int, Dictionary<string, float>> LoadScores(string path)
    {
        var scores = new Dictionary<int, Dictionary<string, float>>();
        foreach (var line in File.ReadLines(path).Where(l => l.Length > 0))
        {
            using var pair = JsonDocument.Parse(line);
            var index = pair.RootElement.GetProperty("CacheIndex").GetInt32();
            if (!scores.TryGetValue(index, out var tags))
                scores[index] = tags = new Dictionary<string, float>(StringComparer.Ordinal);
            tags[pair.RootElement.GetProperty("Tag").GetString()!.Replace(' ', '_')] = pair.RootElement.GetProperty("Full").GetSingle();
        }
        return scores;
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
