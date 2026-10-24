using UnbooruTagger.Core.Scoring;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class TagRankingEvaluatorTests
{
    [Fact]
    public void RankingMetrics_PerfectSeparation_HasPerfectApAndAThresholdBetweenTheGroups()
    {
        var (ap, f1, threshold) = TagRankingEvaluator.RankingMetrics([0.9f, 0.8f], [0.3f, 0.2f, 0.1f]);

        Assert.Equal(1.0, ap, 6);
        Assert.Equal(1.0, f1, 6);
        Assert.InRange(threshold, 0.3f, 0.8f);
    }

    [Fact]
    public void RankingMetrics_NegativeAboveEveryPositive_LowersAp()
    {
        var (ap, _, _) = TagRankingEvaluator.RankingMetrics([0.5f, 0.4f], [0.9f, 0.1f]);

        // Ranked: neg, pos, pos, neg -> precisions at the positives 1/2 and 2/3.
        Assert.Equal((0.5 + 2.0 / 3) / 2, ap, 6);
    }

    [Fact]
    public void RankingMetrics_TiesArePessimistic()
    {
        var (ap, _, _) = TagRankingEvaluator.RankingMetrics([0.5f], [0.5f]);

        Assert.Equal(0.5, ap, 6);
    }

    [Fact]
    public void Evaluate_SeparableTags_ScorePerfectlyAndCalibrateAThresholdBetweenTheGroups()
    {
        // Tag 0 lives on dimension 0, tag 1 on dimension 1; each image points at its tag.
        const int dim = 2;
        var imageTags = new List<IReadOnlyList<int>>();
        var pooled = new List<float>();
        for (var i = 0; i < 60; i++)
        {
            var tag = i % 3 == 0 ? 0 : 1;
            imageTags.Add([tag]);
            pooled.AddRange(tag == 0 ? [1f, 0f] : [0f, 1f]);
        }

        var vocabulary = TagVocabulary.CreateEmpty();
        vocabulary.AddTag("a");
        vocabulary.AddTag("b");
        var statistics = TagStatistics.Build(imageTags, 2, samplesPerTag: 1000);

        var report = TagRankingEvaluator.Evaluate(
            pooled.ToArray(), imageTags, [1f, 0f, 0f, 1f], dim, new LogitScaling(10f, -5f), vocabulary, statistics,
            ExclusiveTagGroups.Empty, CPU, minPositives: 5);

        Assert.Equal(2, report.EvaluatedTags);
        Assert.Equal(1.0, report.MeanAveragePrecision, 6);
        Assert.Equal(1.0, report.MeanHeldOutF1, 6);
        Assert.Equal(1.0, report.MicroPrecision, 6);
        Assert.Equal(1.0, report.MicroRecall, 6);
        foreach (var tag in report.Tags)
            Assert.InRange(tag.Threshold, 1f / (1 + MathF.Exp(5f)), 1f / (1 + MathF.Exp(-5f))); // between sigmoid(-5) and sigmoid(5)
    }
}
