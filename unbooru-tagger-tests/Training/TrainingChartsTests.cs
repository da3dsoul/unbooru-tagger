using SkiaSharp;
using UnbooruTagger.Training.Commands;
using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class TrainingChartsTests
{
    [Fact]
    public void Ticks_AreRoundValuesInsideTheRange()
    {
        Assert.Equal([0.2, 0.4, 0.6, 0.8], TrainingCharts.Ticks(0.15, 0.85, 4));
        Assert.Equal([50_000d, 100_000, 150_000], TrainingCharts.Ticks(24_946, 165_000, 3));
    }

    [Fact]
    public void Render_DrawsAnImageOfTheRequestedSize_WithOrWithoutData()
    {
        var empty = new TrainingSeries("run", new LossHistory(), [], 0, 1000, 6, []);
        using var blank = TrainingCharts.Render(empty, 800, 500, DateTime.Now);
        Assert.Equal((800, 500), (blank.Width, blank.Height));

        var loss = new LossHistory(bucketSteps: 10);
        for (var step = 1; step <= 100; step++)
            loss.Add(step, 2f - step / 100f);
        EvaluationPoint Point(long step, double map) => new(step, step % 1000 == 0, map, 0.3, 0.05, 0.1, 0.9, 0.2);
        var series = new TrainingSeries("run", loss, [Point(500, 0.2), Point(1000, 0.4), Point(1500, 0.5)], 1600, 1000, 6,
            [("other run", Point(0, 0.7))],
            [("compared", new TrainingSeries("compared", loss, [Point(400, 0.3), Point(1200, 0.45), Point(9000, 0.9)], 9000, 1000, 6, []))]);
        using var bitmap = TrainingCharts.Render(series, 1200, 700, DateTime.Now);
        Assert.Equal((1200, 700), (bitmap.Width, bitmap.Height));
        // The charts drew more than the background and panels.
        Assert.True(bitmap.Pixels.Distinct().Count() > 20);
    }

    [Fact]
    public void Render_DrawsComparedEvaluations_OnlyUpToThisRunsLatestEvaluation()
    {
        var loss = new LossHistory(bucketSteps: 10);
        for (var step = 1; step <= 100; step++)
            loss.Add(step, 2f - step / 100f);
        EvaluationPoint Point(long step, double map) => new(step, false, map, 0.3, 0.05, 0.1, 0.9, 0.2);
        SKBitmap Draw(params EvaluationPoint[] compared) => TrainingCharts.Render(
            new TrainingSeries("run", loss, [Point(500, 0.2), Point(1000, 0.4)], 1600, 1000, 6, [],
                [("compared", new TrainingSeries("compared", loss, compared, 9000, 1000, 6, []))]),
            1200, 700, DateTime.Now);

        // Step 1,500 is before the current step (1,600) but after this run's latest evaluation (1,000): not drawn yet.
        using var without = Draw(Point(400, 0.3), Point(900, 0.45));
        using var with = Draw(Point(400, 0.3), Point(900, 0.45), Point(1500, 0.6));
        Assert.Equal(without.Pixels, with.Pixels);
    }

    [Fact]
    public void Sixel_KeepsUpTo256ColorsExactly_AndRunLengthEncodesRows()
    {
        using var bitmap = new SKBitmap(10, 7);
        for (var y = 0; y < 7; y++)
            for (var x = 0; x < 10; x++)
                bitmap.SetPixel(x, y, x < 5 ? new SKColor(200, 10, 10) : new SKColor(10, 10, 200));

        var (palette, indices) = Sixel.Quantize(bitmap.Pixels);
        Assert.Equal(2, palette.Length);
        Assert.Contains(new SKColor(200, 10, 10), palette);
        Assert.Equal(palette[indices[0]], new SKColor(200, 10, 10));

        var sixel = Sixel.Encode(bitmap);
        Assert.StartsWith("\u001bP0;1;0q\"1;1;10;7", sixel);
        Assert.EndsWith("\u001b\\", sixel);
        // Two bands (7 rows): the first is full columns ('~', all six bits) five wide per color.
        var red = Array.IndexOf(palette, new SKColor(200, 10, 10));
        Assert.Contains($"#{red}!5~", sixel);
        Assert.Equal(2, sixel.Count(c => c == '-'));
    }

    [Fact]
    public void Series_PutsEachEpochEvaluationAtTheEndOfItsEpoch_LabelledWithThatEpoch()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            // Run 4's layout: 484,921 steps per epoch, 26,846 steps into epoch 4.
            File.WriteAllText(Path.Combine(directory, "training_progress.json"),
                """{"CompletedEpochs":3,"EarlyStoppingBestLoss":1,"EarlyStoppingEvaluationsSinceImprovement":0,"CompletedStepsInEpoch":26846,"ValidationSplitSeed":1,"StepsPerEpoch":484921,"EarlyStoppingMetric":"1-mAP"}""");
            for (var e = 1; e <= 3; e++)
            {
                var dir = Directory.CreateDirectory(Path.Combine(directory, "history", $"epoch-{e}")).FullName;
                File.WriteAllText(Path.Combine(dir, "evaluation_report.json"), $$"""{"MeanAveragePrecision":0.{{e}},"ExclusiveViolationRate":0.1}""");
            }

            var series = TrainingSeries.Load(directory, epochs: 6);
            Assert.Equal([(484_921L, 0.1), (969_842L, 0.2), (1_454_763L, 0.3)],
                series.Points.Select(p => (p.Step, Math.Round(p.MeanAveragePrecision, 6))));
            Assert.All(series.Points, p => Assert.True(p.EndOfEpoch));
            // Each boundary is named for the epoch ending there, so epoch 3's label sits on epoch 3's evaluation.
            Assert.Equal([(484_921d, "epoch 1"), (969_842d, "epoch 2"), (1_454_763d, "epoch 3")],
                TrainingCharts.EpochEnds(series, 0, series.CurrentStep));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Series_ReadsSnapshotsUpToTheCurrentStep()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            File.WriteAllText(Path.Combine(directory, "training_progress.json"),
                """{"CompletedEpochs":0,"EarlyStoppingBestLoss":1,"EarlyStoppingEvaluationsSinceImprovement":0,"CompletedStepsInEpoch":25000,"ValidationSplitSeed":1,"StepsPerEpoch":100000,"EarlyStoppingMetric":"1-mAP"}""");
            foreach (var (step, done) in new[] { (10_000, true), (20_000, false), (30_000, true) })
            {
                var dir = Directory.CreateDirectory(Path.Combine(directory, "snapshots", $"step-{step}")).FullName;
                File.WriteAllText(Path.Combine(dir, "evaluation_report.json"), $$"""{"MeanAveragePrecision":0.{{step / 10_000}},"ExclusiveViolationRate":0.1}""");
                if (done)
                    File.WriteAllText(Path.Combine(dir, "evaluation.done"), "exit code 0");
            }

            var series = TrainingSeries.Load(directory, epochs: 6);
            // 20K is still being evaluated; 30K is past the latest checkpoint (another timeline).
            var point = Assert.Single(series.Points);
            Assert.Equal(10_000, point.Step);
            Assert.Equal(0.1, point.MeanAveragePrecision, 6);
            Assert.Equal(25_000, series.CurrentStep);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
