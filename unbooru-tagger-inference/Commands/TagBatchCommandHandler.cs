using System.Text.Json;
using Spectre.Console;
using UnbooruTagger.Core.Runtime;

namespace UnbooruTagger.Inference.Commands;

public static class TagBatchCommandHandler
{
    private static readonly string[] SupportedExtensions = [".png", ".jpg", ".jpeg", ".webp", ".bmp"];

    public static int Run(string modelDir, string directory, float threshold)
    {
        using var model = ModelBundle.Load(modelDir);

        var images = Directory.EnumerateFiles(directory)
            .Where(path => SupportedExtensions.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase))
            .OrderBy(path => path, StringComparer.Ordinal)
            .ToList();

        var output = new Dictionary<string, object>();
        AnsiConsole.Progress()
            .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(), new PercentageColumn(), new RemainingTimeColumn())
            .Start(ctx =>
            {
                var task = ctx.AddTask("Tagging images", maxValue: images.Count);
                foreach (var imagePath in images)
                {
                    task.Description = $"Tagging {Path.GetFileName(imagePath)}";
                    var scores = TagCommandHandler.Score(model, imagePath, threshold);
                    output[Path.GetFileName(imagePath)] = scores.ToDictionary(r => r.Tag, r => r.Confidence);
                    task.Increment(1);
                }
            });

        Console.WriteLine(JsonSerializer.Serialize(output, new JsonSerializerOptions { WriteIndented = true }));
        return 0;
    }
}
