using System.Text.Json;
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
            .OrderBy(path => path, StringComparer.Ordinal);

        var output = new Dictionary<string, object>();
        foreach (var imagePath in images)
        {
            var scores = TagCommandHandler.Score(model, imagePath, threshold);
            output[Path.GetFileName(imagePath)] = scores.ToDictionary(r => r.Tag, r => r.Confidence);
        }

        Console.WriteLine(JsonSerializer.Serialize(output, new JsonSerializerOptions { WriteIndented = true }));
        return 0;
    }
}
