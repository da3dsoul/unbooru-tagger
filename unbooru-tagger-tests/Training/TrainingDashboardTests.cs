using Spectre.Console;
using UnbooruTagger.Training.Commands;
using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class TrainingDashboardTests
{
    private static readonly TrainingRunInfo Info = new("training · test", Epochs: 6, StepsPerEpoch: 484_921, StartEpoch: 0, StartStepInEpoch: 199_000,
        ImagesPerStep: 10, CheckpointInterval: TimeSpan.FromMinutes(15));

    [Fact]
    public void Rate_IsMeasuredOverTheTrailingWindow_NotTheWholeRun()
    {
        var now = new DateTime(2026, 10, 22, 12, 0, 0);
        var dashboard = new TrainingDashboard(Info, () => now);

        // 30 minutes at 2 steps/s, then 30 minutes at 4 steps/s: the trailing rate tracks the new speed.
        var step = 199_000;
        for (var s = 0; s < 3600; s++)
        {
            now = now.AddSeconds(1);
            step += s < 1800 ? 2 : 4;
            dashboard.StepCompleted(step, 1f, 1f, 0f, 1e-4, 250, 0);
        }

        Assert.InRange(dashboard.StepsPerSecond(), 3.9, 4.1);
    }

    [Fact]
    public void StepProgress_MovesTheStepEveryStep_AndLossesArriveLaterAtTheirOwnSteps()
    {
        // Training reports each step as it ends and its losses when it reads them back, every 25 steps.
        var now = new DateTime(2026, 10, 22, 12, 0, 0);
        var dashboard = new TrainingDashboard(Info, () => now);
        for (var step = 199_001; step <= 199_025; step++)
        {
            now = now.AddSeconds(0.2);
            dashboard.StepProgress(step, 1e-4, 200, 0);
        }
        Assert.Equal(199_025, dashboard.ChartData().Series.CurrentStep);

        for (var step = 199_001; step <= 199_025; step++)
            dashboard.StepLosses(step, 3f, 2f, 1f);
        var history = dashboard.ChartData().Series.Loss;
        Assert.InRange(history.Buckets.Count == 0 ? 0 : history.Buckets[^1].Mean, 0, 3f);
        Assert.InRange(dashboard.StepsPerSecond(), 4.9, 5.1);
    }

    [Fact]
    public void Render_ShowsEpochAndRunEstimatesSeparately()
    {
        var now = new DateTime(2026, 10, 22, 12, 0, 0);
        var dashboard = new TrainingDashboard(Info, () => now);
        var step = 199_000;
        for (var s = 0; s < 4000; s++)
        {
            now = now.AddSeconds(0.25);
            dashboard.StepCompleted(++step, 2f - s / 4000f, 1.5f, 0.3f, 9.6e-5, 270, 0);
        }

        var writer = new StringWriter();
        var console = AnsiConsole.Create(new AnsiConsoleSettings
        {
            Ansi = AnsiSupport.No, ColorSystem = ColorSystemSupport.NoColors, Out = new AnsiConsoleOutput(writer), Interactive = InteractionSupport.No
        });
        console.Profile.Width = 237;
        console.Write(dashboard.Render());
        var text = writer.ToString();

        // 4 steps/s: (484,921 - 203,000) steps left in the epoch ≈ 19h 34m.
        Assert.Contains("19h 34m left · epoch ends", text);
        Assert.Contains("epoch 1/6", text);
        Assert.Contains("finishes", text);
        Assert.Contains("loss, mean per 2,000 steps", text);
        Assert.Contains("none yet: runs at the end of epoch 1", text);
        Assert.DoesNotContain("…", text.Replace("estimating…", ""));
    }

    [Fact]
    public void Formatting()
    {
        Assert.Equal("8d 13h", TrainingDashboard.Duration(TimeSpan.FromHours(205)));
        Assert.Equal("21h 05m", TrainingDashboard.Duration(new TimeSpan(21, 5, 0)));
        Assert.Equal("4m 05s", TrainingDashboard.Duration(new TimeSpan(0, 4, 5)));

        var now = new DateTime(2026, 10, 22, 12, 0, 0);
        Assert.Equal("23:12", TrainingDashboard.Clock(new DateTime(2026, 10, 22, 23, 12, 0), now));
        Assert.Equal("Fri 02:05", TrainingDashboard.Clock(new DateTime(2026, 10, 23, 2, 5, 0), now));
        Assert.Equal("Oct 31 02:05", TrainingDashboard.Clock(new DateTime(2026, 10, 31, 2, 5, 0), now));

        Assert.Equal("█▁▅", TrainingDashboard.Sparkline([3f, 1f, 2f]));
        Assert.Equal("▁▁", TrainingDashboard.Sparkline([1f, 1f]));
    }

    [Fact]
    public void Evaluation_IsShownAndItsDurationAddedToTheRunEstimate()
    {
        var now = new DateTime(2026, 10, 22, 12, 0, 0);
        var dashboard = new TrainingDashboard(Info with { StartStepInEpoch = 0 }, () => now);
        dashboard.EvaluationStarted();
        now = now.AddMinutes(10);
        dashboard.EvaluationFinished(0, new EvaluationReport(100, 50, 0.81, 0.85, 0.5, 0.45, 0.4, 0.42, 0.41, 0.5, 0.01, []), null);

        var writer = new StringWriter();
        var console = AnsiConsole.Create(new AnsiConsoleSettings { Ansi = AnsiSupport.No, Out = new AnsiConsoleOutput(writer) });
        console.Profile.Width = 237;
        console.Write(dashboard.Render());

        Assert.Contains("epoch 1: mAP 0.8100", writer.ToString());
        Assert.Contains("incl. 10m 00s evaluation per epoch", writer.ToString());
    }

    [Fact]
    public void OffloadedEvaluation_IsShownWhileRunningAndReplacedByItsResult()
    {
        var now = new DateTime(2026, 10, 25, 0, 17, 0);
        var dashboard = new TrainingDashboard(Info with { StartEpoch = 1, StartStepInEpoch = 0, EvaluationsOffloaded = true }, () => now);
        dashboard.EvaluationOffloaded(1);
        now = now.AddMinutes(12);

        var running = Render(dashboard);
        Assert.Contains("epoch 1 in another process since 00:17, running 12m 00s", running);
        Assert.Contains("none applied yet", running);
        Assert.Contains("run in another process alongside training", running);
        Assert.DoesNotContain("none yet: runs at the end of epoch", running);

        dashboard.EvaluationFinished(0, new EvaluationReport(100, 50, 0.81, 0.85, 0.5, 0.45, 0.4, 0.42, 0.41, 0.5, 0.01, []), null);
        var applied = Render(dashboard);
        Assert.DoesNotContain("in another process since", applied);
        Assert.Contains("epoch 1: mAP 0.8100", applied);
    }

    [Fact]
    public void EvaluationPoints_AreChartedWithTheirLatestChange()
    {
        var now = new DateTime(2026, 10, 27, 2, 0, 0);
        var dashboard = new TrainingDashboard(Info with { EvaluationsOffloaded = true, SnapshotEvaluationSteps = 10_000 }, () => now);
        dashboard.AddEvaluationPoint(new EvaluationPoint(190_000, false, 0.40, 0.29, 0.056, 0.107, 0.86, 0.27, ObjectEffect: 0.41));
        dashboard.AddEvaluationPoint(new EvaluationPoint(200_000, false, 0.42, 0.30, 0.060, 0.100, 0.84, double.NaN, ObjectEffect: 0.45));
        dashboard.SnapshotEvaluationStarted(210_000);

        var text = Render(dashboard);
        Assert.Contains("every 10,000 steps · 2 evaluated so far · next at step 200,000", text);
        Assert.Contains("evaluating step 210,000 (since 02:00)", text);
        Assert.Contains("held-out mAP 0.420 +0.020 since step 190K", text);
        Assert.Contains("object effect (log-odds) 0.450 +0.040 since step 190K", text);
        Assert.Contains("context recall gap 6.0% +0.4% since step 190K ↓", text);
        Assert.Contains("exclusive-tag violations 27.0%", text); // the latest point without it keeps the last known value
        Assert.Contains("190K", text);
        Assert.Contains("200K", text);
        Assert.DoesNotContain("…", text.Replace("estimating…", "")); // nothing cut off
    }

    private static string Render(TrainingDashboard dashboard)
    {
        var writer = new StringWriter();
        var console = AnsiConsole.Create(new AnsiConsoleSettings { Ansi = AnsiSupport.No, Out = new AnsiConsoleOutput(writer) });
        console.Profile.Width = 237;
        console.Write(dashboard.Render());
        return writer.ToString();
    }

    [Fact]
    public void ChartData_DrawsComparedRuns_AndRereadsThemEveryTenMinutes()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            File.WriteAllText(Path.Combine(directory, "training_progress.json"),
                """{"CompletedEpochs":0,"EarlyStoppingBestLoss":1,"EarlyStoppingEvaluationsSinceImprovement":0,"CompletedStepsInEpoch":300000,"ValidationSplitSeed":1,"StepsPerEpoch":484921,"EarlyStoppingMetric":"1-mAP"}""");
            void Snapshot(int step)
            {
                var dir = Directory.CreateDirectory(Path.Combine(directory, "snapshots", $"step-{step}")).FullName;
                File.WriteAllText(Path.Combine(dir, "evaluation_report.json"), """{"MeanAveragePrecision":0.5,"ExclusiveViolationRate":0.1}""");
                File.WriteAllText(Path.Combine(dir, "evaluation.done"), "exit code 0");
            }
            Snapshot(100_000);
            var now = new DateTime(2026, 10, 31, 17, 0, 0);
            var dashboard = new TrainingDashboard(Info, () => now);
            dashboard.SetChartComparisons([("run 4", directory)]);

            var (series, version) = dashboard.ChartData();
            var (label, run) = Assert.Single(series.Comparisons!);
            Assert.Equal("run 4", label);
            Assert.Single(run.Points);

            // A new evaluation in the compared run shows up at the next re-read, not before.
            Snapshot(200_000);
            now = now.AddMinutes(5);
            Assert.Equal(version, dashboard.ChartData().Version);
            now = now.AddMinutes(6);
            var (later, laterVersion) = dashboard.ChartData();
            Assert.Equal(2, later.Comparisons![0].Run.Points.Count);
            Assert.NotEqual(version, laterVersion);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
