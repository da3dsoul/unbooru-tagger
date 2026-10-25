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
        Assert.Contains("trend", text);
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

    private static string Render(TrainingDashboard dashboard)
    {
        var writer = new StringWriter();
        var console = AnsiConsole.Create(new AnsiConsoleSettings { Ansi = AnsiSupport.No, Out = new AnsiConsoleOutput(writer) });
        console.Profile.Width = 237;
        console.Write(dashboard.Render());
        return writer.ToString();
    }
}
