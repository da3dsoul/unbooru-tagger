using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class ScaleStatsTests
{
    [Fact]
    public void FittedMilliseconds_RecoverEachSizesStepTime_FromMixedIntervals()
    {
        var scales = TrainingScales.Parse("512:10:1,1024:3:1:0.5,2048:1:1:1");
        var stats = new ScaleStats(scales);
        var truth = new Dictionary<int, double> { [512] = 140, [1024] = 190, [2048] = 400 };
        var random = new Random(3);
        for (var interval = 0; interval < 200; interval++)
        {
            double ms = 0;
            for (var k = 0; k < 5; k++)
            {
                var scale = scales[random.Next(scales.Count)];
                stats.Step(scale.Size, scale.Batch, fromStore: scale.Size == 512 ? 0 : 1, drawnStored: scale.Size == 2048 ? 1 : 0);
                stats.Loss(scale.Size, scale.Size / 1000.0);
                ms += truth[scale.Size];
            }
            stats.Interval(ms, deviceMb: 1000 + interval);
        }
        stats.Interval(60_000, deviceMb: 0); // a checkpoint save: dropped from the fit

        var fitted = stats.FittedMilliseconds();
        Assert.Equal(140, fitted[0], 3);
        Assert.Equal(190, fitted[1], 3);
        Assert.Equal(400, fitted[2], 3);
        var summary = stats.TakeSummary("stores 1024px 5");
        Assert.NotNull(summary);
        Assert.Contains("1024px", summary);
        Assert.Contains("loss 1.024", summary);
        Assert.Contains("VRAM peak 1.2 GB", summary); // 1,199 MB
        Assert.Null(stats.TakeSummary("stores"));
    }

    [Fact]
    public void ASizeNeverDrawn_IsReportedSkipped()
    {
        var stats = new ScaleStats(TrainingScales.Parse("512:10:0.9,2048:1:0.1:1"));
        for (var k = 0; k < 10; k++)
            stats.Step(512, 10, 0, 0);
        stats.Interval(1400, 0);

        Assert.Contains("2048px skipped (plan 10%)", stats.TakeSummary("no stores"));
    }
}
