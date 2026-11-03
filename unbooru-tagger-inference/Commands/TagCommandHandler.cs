using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Runtime;
using UnbooruTagger.Core.Scoring;

namespace UnbooruTagger.Inference.Commands;

public static class TagCommandHandler
{
    public static int Run(string modelDir, string imagePath, float? threshold)
    {
        using var model = ModelBundle.Load(modelDir);

        var scores = Score(model, imagePath, threshold);
        if (PredictRating(model, scores) is { } rating)
            Console.WriteLine($"rating\t{rating}");
        foreach (var (tag, confidence) in scores)
            Console.WriteLine($"{confidence:F3}\t{tag}");

        return 0;
    }

    /// <summary>
    /// The content rating implied by the cumulative rating labels among <paramref name="reported"/>
    /// (the highest one reported wins), or null if the model wasn't trained with ratings.
    /// </summary>
    public static ContentRating? PredictRating(ModelBundle model, IReadOnlyList<(string Tag, float Confidence)> reported)
    {
        if (!ImageRatings.CumulativeTags.All(t => model.Vocabulary.TryGet(t, out _)))
            return null;
        var level = 0;
        for (var k = 0; k < ImageRatings.CumulativeTags.Length; k++)
            if (reported.Any(r => r.Tag == ImageRatings.CumulativeTags[k]))
                level = k + 1;
        return (ContentRating)level;
    }

    /// <param name="threshold">An explicit cutoff for every tag, or null to use each tag's calibrated threshold (see <see cref="ModelBundle.ThresholdFor"/>).</param>
    public static List<(string Tag, float Confidence)> Score(ModelBundle model, string imagePath, float? threshold) =>
        Score(model, model.ImageEncoder.Encode(imagePath), threshold);

    public static List<(string Tag, float Confidence)> Score(ModelBundle model, ImageEncoding encoding, float? threshold) =>
        ScoreRows(model, encoding, threshold)
            .Select(r => (model.Vocabulary.GetByRowIndex(r.Row).Tag, r.Confidence))
            .ToList();

    /// <summary>
    /// Vocabulary rows that clear their threshold, strongest first, with contradictory pairs resolved. Given
    /// <paramref name="scores"/>, only its tags are scored, with its confidences instead of the bundle's (scores the bundle
    /// can't compute, see render-gold --scores: the bundle's own scores for the rest are then meaningless, and thousands of
    /// them clearing their thresholds each cost a heatmap and boxes). Given <paramref name="onlyTags"/> (underscored, as booru
    /// tags are written), only those tags are scored.
    /// </summary>
    internal static List<(int Row, float Confidence)> ScoreRows(
        ModelBundle model, ImageEncoding encoding, float? threshold, IReadOnlyDictionary<string, float>? scores = null,
        IReadOnlySet<string>? onlyTags = null)
    {
        var results = new List<(int Row, float Confidence)>();
        for (var row = 0; row < model.Embeddings.RowCount; row++)
        {
            if (onlyTags is not null && !onlyTags.Contains(model.Vocabulary.GetByRowIndex(row).Tag.Replace(' ', '_')))
                continue;
            float confidence;
            if (scores is null)
                confidence = TagScorer.Score(encoding.PooledEmbedding, model.Embeddings.GetRow(row), model.LogitScaling);
            else if (!scores.TryGetValue(model.Vocabulary.GetByRowIndex(row).Tag.Replace(' ', '_'), out confidence))
                continue;
            if (confidence >= model.ThresholdFor(row, threshold))
                results.Add((row, confidence));
        }

        results.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
        return model.ExclusiveGroups.Resolve(results, r => r.Row).ToList();
    }
}
