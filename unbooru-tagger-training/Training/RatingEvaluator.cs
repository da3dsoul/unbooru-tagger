using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Scoring;

namespace UnbooruTagger.Training.Training;

/// <param name="Thresholds">Per cumulative label (≥sensitive, ≥questionable, explicit): the cutoff reaching the target recall.</param>
/// <param name="Recall">Held-out recall of each cumulative label at its cutoff.</param>
/// <param name="Precision">Held-out precision of each cumulative label at its cutoff.</param>
/// <param name="Confusion">[true rating, predicted rating] image counts, General..Explicit.</param>
/// <param name="ExplicitOrQuestionableCalledGeneral">Fraction of questionable/explicit images predicted general — the costly mistake.</param>
/// <param name="GeneralCalledExplicitOrQuestionable">Fraction of general images predicted questionable/explicit — the cautious mistake.</param>
public sealed record RatingReport(
    int Images, float[] Thresholds, double[] Recall, double[] Precision, long[][] Confusion,
    double Accuracy, double ExplicitOrQuestionableCalledGeneral, double GeneralCalledExplicitOrQuestionable);

/// <summary>
/// Calibrates and scores the cumulative rating labels on held-out images against their booru
/// ratings. Cutoffs are chosen for recall, not F1: each label's cutoff is the highest one that
/// still catches <c>targetRecall</c> of the images that truly carry it, because missing an
/// explicit image costs far more than flagging a borderline one. The predicted rating is the
/// highest cumulative label that clears its cutoff.
/// </summary>
public static class RatingEvaluator
{
    public static RatingReport Evaluate(
        float[] pooled, int dim, IReadOnlyList<ContentRating> ratings, float[] tagEmbeddings, int[] cumulativeRows,
        LogitScaling scaling, double[] targetRecall)
    {
        var known = Enumerable.Range(0, ratings.Count).Where(i => ratings[i] != ContentRating.Unknown).ToArray();
        var scores = new float[3][];
        for (var k = 0; k < 3; k++)
        {
            var row = cumulativeRows[k];
            scores[k] = known.Select(i =>
            {
                var sum = 0f;
                for (var d = 0; d < dim; d++)
                    sum += pooled[i * dim + d] * tagEmbeddings[row * dim + d];
                return 1f / (1f + MathF.Exp(-scaling.Apply(sum)));
            }).ToArray();
        }

        var thresholds = new float[3];
        var recall = new double[3];
        var precision = new double[3];
        for (var k = 0; k < 3; k++)
        {
            var positives = known.Select((image, j) => (Score: scores[k][j], Positive: (int)ratings[image] > k)).ToArray();
            var positiveScores = positives.Where(p => p.Positive).Select(p => p.Score).OrderByDescending(s => s).ToArray();
            if (positiveScores.Length == 0)
            {
                thresholds[k] = 0.5f;
                continue;
            }

            var keep = Math.Clamp((int)Math.Ceiling(targetRecall[k] * positiveScores.Length), 1, positiveScores.Length);
            thresholds[k] = positiveScores[keep - 1];
            var predicted = positives.Count(p => p.Score >= thresholds[k]);
            var truePositives = positives.Count(p => p.Positive && p.Score >= thresholds[k]);
            recall[k] = truePositives / (double)positiveScores.Length;
            precision[k] = predicted > 0 ? truePositives / (double)predicted : 0;
        }

        var confusion = Enumerable.Range(0, 4).Select(_ => new long[4]).ToArray();
        for (var j = 0; j < known.Length; j++)
        {
            var level = 0;
            for (var k = 0; k < 3; k++)
                if (scores[k][j] >= thresholds[k])
                    level = k + 1;
            confusion[(int)ratings[known[j]]][level]++;
        }

        var total = known.Length;
        var correct = Enumerable.Range(0, 4).Sum(r => confusion[r][r]);
        var risky = confusion[2].Sum() + confusion[3].Sum();
        var general = confusion[0].Sum();
        return new RatingReport(
            total, thresholds, recall, precision, confusion,
            total > 0 ? correct / (double)total : 0,
            risky > 0 ? (confusion[2][0] + confusion[3][0]) / (double)risky : 0,
            general > 0 ? (confusion[0][2] + confusion[0][3]) / (double)general : 0);
    }
}
