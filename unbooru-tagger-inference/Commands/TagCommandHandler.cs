using UnbooruTagger.Core.Runtime;
using UnbooruTagger.Core.Scoring;

namespace UnbooruTagger.Inference.Commands;

public static class TagCommandHandler
{
    public static int Run(string modelDir, string imagePath, float threshold)
    {
        using var model = ModelBundle.Load(modelDir);

        foreach (var (tag, confidence) in Score(model, imagePath, threshold))
            Console.WriteLine($"{confidence:F3}\t{tag}");

        return 0;
    }

    public static List<(string Tag, float Confidence)> Score(ModelBundle model, string imagePath, float threshold)
    {
        var encoding = model.ImageEncoder.Encode(imagePath);

        var results = new List<(string Tag, float Confidence)>();
        for (var row = 0; row < model.Embeddings.RowCount; row++)
        {
            var confidence = TagScorer.Score(encoding.PooledEmbedding, model.Embeddings.GetRow(row));
            if (confidence >= threshold)
                results.Add((model.Vocabulary.GetByRowIndex(row).Tag, confidence));
        }

        results.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
        return results;
    }
}
