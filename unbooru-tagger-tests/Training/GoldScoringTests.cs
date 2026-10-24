using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class GoldScoringTests
{
    private static GoldInstance Box(GoldKind kind, int x, int y, int w, int h) => new("tag", [x, y, w, h], kind);

    [Fact]
    public void CertainBoxes_ScoreMatchesMissesAndFalsePositives()
    {
        GoldInstance[] gold = [Box(GoldKind.Certain, 0, 0, 10, 10), Box(GoldKind.Certain, 50, 50, 10, 10)];
        int[][] extracted = [[0, 0, 10, 10], [80, 80, 10, 10]]; // one match, one false positive; second gold missed

        var score = LocalizationEvaluator.ScorePair(gold, (5, 5), 0.5, extracted);

        Assert.Equal(true, score.PeakHit);
        Assert.Equal((2, 1), (score.ScoredExtracted, score.MatchedExtracted));
        Assert.Equal((2, 1), (score.RecallItems, score.RecalledItems));
        Assert.Equal(0, score.CountError);
    }

    [Fact]
    public void GroupBox_IsFoundByAnythingInside_AndNothingInsideCountsAgainst()
    {
        GoldInstance[] gold = [Box(GoldKind.Group, 0, 0, 100, 100)];
        int[][] extracted = [[10, 10, 5, 5], [40, 40, 5, 5], [30, 30, 8, 8]];

        var score = LocalizationEvaluator.ScorePair(gold, (12, 12), 0.5, extracted);

        Assert.Equal(true, score.PeakHit);
        Assert.Equal(0, score.ScoredExtracted);
        Assert.Equal((1, 1), (score.RecallItems, score.RecalledItems));
        Assert.Null(score.CountError);
    }

    [Fact]
    public void IgnoreRegion_ExcusesPredictionsAndNeutralizesThePeak()
    {
        GoldInstance[] gold = [Box(GoldKind.Certain, 0, 0, 10, 10), Box(GoldKind.Ignore, 50, 50, 20, 20)];
        int[][] extracted = [[52, 52, 10, 10]];

        var score = LocalizationEvaluator.ScorePair(gold, (55, 55), 0.5, extracted);

        Assert.Null(score.PeakHit);
        Assert.Equal(0, score.ScoredExtracted);
        Assert.Equal((1, 0), (score.RecallItems, score.RecalledItems));
        Assert.Null(score.CountError);
    }
}
