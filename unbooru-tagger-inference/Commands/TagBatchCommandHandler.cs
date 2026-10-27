using System.Text.Json;
using Spectre.Console;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Runtime;

namespace UnbooruTagger.Inference.Commands;

public static class TagBatchCommandHandler
{
    public static int Run(string modelDir, string directory, float? threshold, int batchSize = 16, int workers = 0, InferenceDevice device = InferenceDevice.Auto)
    {
        using var model = ModelBundle.Load(modelDir, device: device);
        var images = BatchPipeline.FindImages(directory, recursive: false);
        workers = workers > 0 ? workers : Math.Max(1, Environment.ProcessorCount / 2);

        var output = new SortedDictionary<string, object>(StringComparer.Ordinal);
        var failures = 0;
        AnsiConsole.Progress()
            .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(), new PercentageColumn(), new RemainingTimeColumn())
            .Start(ctx =>
            {
                var task = ctx.AddTask("Tagging images", maxValue: images.Count);
                BatchPipeline.Run(model, images, batchSize, workers, withBoxes: false, keepBitmaps: false,
                    (item, encoding) =>
                    {
                        var scores = TagCommandHandler.Score(model, encoding, threshold);
                        lock (output)
                            output[Path.GetFileName(item.Path)] = scores.ToDictionary(r => r.Tag, r => r.Confidence);
                        task.Increment(1);
                    },
                    (path, e) =>
                    {
                        Interlocked.Increment(ref failures);
                        AnsiConsole.MarkupLineInterpolated($"[red]{path}: {e.GetType().Name}: {e.Message}[/]");
                        task.Increment(1);
                    });
            });

        Console.WriteLine(JsonSerializer.Serialize(output, new JsonSerializerOptions { WriteIndented = true }));
        return failures > 0 ? 1 : 0;
    }
}
