using System.Text.Json;
using SkiaSharp;
using Spectre.Console;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Runtime;
using UnbooruTagger.Core.Scoring;

namespace UnbooruTagger.Inference.Commands;

/// <summary>
/// Tags and detects every image in a directory in one pass (<see cref="BatchPipeline"/>): one
/// model run per image gives both the whole-image tags and, for localizable tags, their boxes.
/// Writes one JSON object per line to <c>--out</c>, in completion order:
/// <c>{"image": path relative to the directory, "rating": ..., "tags": [{"tag", "confidence", "boxes": [...]}]}</c>.
/// </summary>
public static class DetectBatchCommandHandler
{
    private static readonly JsonSerializerOptions LineOptions = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    public sealed record ImageResult(string Image, string? Rating, IReadOnlyList<TagDetection> Tags);

    public static int Run(
        string modelDir, string directory, string outputPath, string? renderDirectory, bool boxes, bool recursive,
        int batchSize, int workers, InferenceDevice device, float? threshold, float boxThreshold, float boxPercentile,
        float minHeatmapRange, float minBoxFraction, int maxBoxesPerTag)
    {
        using var model = ModelBundle.Load(modelDir, device: device);
        var images = BatchPipeline.FindImages(directory, recursive);
        if (renderDirectory is not null)
            Directory.CreateDirectory(renderDirectory);
        workers = workers > 0 ? workers : Math.Max(1, Environment.ProcessorCount / 2);

        AnsiConsole.MarkupLineInterpolated(
            $"{images.Count:N0} images · model on {((OnnxImageEncoder)model.ImageEncoder).Provider} · batch {batchSize} · {workers} workers{(boxes ? "" : " · tags only")}");

        using var output = new StreamWriter(outputPath);
        var failures = 0;
        var started = DateTime.Now;
        AnsiConsole.Progress()
            .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(), new PercentageColumn(), new ElapsedTimeColumn(), new RemainingTimeColumn())
            .Start(ctx =>
            {
                var task = ctx.AddTask(boxes ? "Tagging and detecting" : "Tagging", maxValue: images.Count);
                BatchPipeline.Run(model, images, batchSize, workers, withBoxes: boxes, keepBitmaps: renderDirectory is not null,
                    (item, encoding) =>
                    {
                        var detections = boxes
                            ? DetectCommandHandler.Detect(model, encoding, item.Guide!, threshold, boxThreshold, boxPercentile, minHeatmapRange, minBoxFraction, maxBoxesPerTag)
                            : TagCommandHandler.Score(model, encoding, threshold).Select(s => new TagDetection(s.Tag, s.Confidence, [])).ToList();
                        var rating = TagCommandHandler.PredictRating(model, detections.Select(d => (d.Tag, d.Confidence)).ToList());
                        var relative = Path.GetRelativePath(directory, item.Path);
                        var line = JsonSerializer.Serialize(new ImageResult(relative, rating?.ToString(), detections), LineOptions);
                        lock (output)
                            output.WriteLine(line);

                        if (renderDirectory is not null && item.Bitmap is not null)
                        {
                            using var annotated = DetectCommandHandler.RenderBoxes(item.Bitmap, detections);
                            using var image = SKImage.FromBitmap(annotated);
                            using var png = image.Encode(SKEncodedImageFormat.Png, quality: 100);
                            using var stream = File.Create(Path.Combine(renderDirectory, Path.ChangeExtension(relative.Replace(Path.DirectorySeparatorChar, '_'), ".png")));
                            png.SaveTo(stream);
                        }
                        task.Increment(1);
                    },
                    (path, e) =>
                    {
                        Interlocked.Increment(ref failures);
                        AnsiConsole.MarkupLineInterpolated($"[red]{path}: {e.GetType().Name}: {e.Message}[/]");
                        task.Increment(1);
                    });
            });

        var seconds = (DateTime.Now - started).TotalSeconds;
        AnsiConsole.MarkupLineInterpolated(
            $"{images.Count - failures:N0} images in {seconds:F1}s ({(images.Count - failures) / seconds:F1} images/s){(failures > 0 ? $", {failures} failed" : "")} → {outputPath}");
        return failures > 0 ? 1 : 0;
    }
}
