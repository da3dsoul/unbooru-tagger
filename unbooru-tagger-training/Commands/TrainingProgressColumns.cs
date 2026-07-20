using Spectre.Console;
using Spectre.Console.Rendering;

namespace UnbooruTagger.Training.Commands;

/// <summary>
/// A description column with a fixed render width instead of Spectre's default of
/// sizing to whatever's currently the longest Description across all rows. Without
/// this, the "Overall" row's changing rate/ETA text and the "Phase" row's changing
/// step/checkpoint/validation text each resize the column every frame as their length
/// changes, so the whole bar/percentage block visibly shifts left and right --
/// unbooru-tagger-data's <c>FixedWidthDescriptionColumn</c> exists for the same reason.
/// </summary>
internal sealed class FixedWidthDescriptionColumn(int width) : ProgressColumn
{
    public override IRenderable Render(RenderOptions options, ProgressTask task, TimeSpan deltaTime) =>
        new Markup(task.Description ?? string.Empty).Overflow(Overflow.Ellipsis);

    public override int? GetColumnWidth(RenderOptions options) => width;
}

/// <summary>
/// Progress-bar rows for <see cref="TrainCommandHandler.Run"/>: "Overall" tracks every
/// step across the whole run (all epochs) with its own rate/ETA computed from the
/// average rate since the run started, folded into the description instead of using
/// Spectre's built-in <c>RemainingTimeColumn</c> -- that column goes blank across the
/// long step-free gaps a checkpoint save or a full validation pass introduces, which
/// reads as "did the ETA break" rather than the run genuinely pausing step progress for
/// a while. "Phase" tracks whatever's happening right now: this epoch's training steps
/// as a real fraction (so a run with a huge total step count -- many epochs times many
/// steps each -- doesn't sit looking stuck at a near-0% overall percentage for the
/// entire first epoch), a checkpoint save (indeterminate, no natural sub-steps), or
/// validation batches (a real fraction).
/// </summary>
internal static class TrainingProgressColumns
{
    private const int DescriptionWidth = 70;

    public static ProgressColumn[] Columns { get; } =
    [
        new FixedWidthDescriptionColumn(DescriptionWidth),
        new ProgressBarColumn(),
        new PercentageColumn(),
        new SpinnerColumn()
    ];

    /// <summary>How often the Overall row's rate/ETA text refreshes on its own, independent of when a step completes.</summary>
    private static readonly TimeSpan OverallRefreshInterval = TimeSpan.FromSeconds(1);

    public static TrainingProgressReporter AddTasks(ProgressContext ctx, int totalSteps, int startingStep)
    {
        var phaseTask = ctx.AddTask("Phase");
        phaseTask.IsIndeterminate = true;

        var overallTask = ctx.AddTask("Overall", maxValue: totalSteps);
        overallTask.Value = startingStep;

        var startedAt = DateTime.UtcNow;
        var latestProcessed = startingStep;

        void RefreshOverallDescription()
        {
            var elapsed = (DateTime.UtcNow - startedAt).TotalSeconds;
            var processed = Volatile.Read(ref latestProcessed);
            var rate = elapsed > 0 ? (processed - startingStep) / elapsed : 0;

            // Average rate since the run started, not a short recent-velocity window:
            // a checkpoint save or validation pass can stall step progress for a while,
            // and a window narrow enough to react quickly to that is exactly narrow
            // enough to spike right after every burst of steps and decay through every
            // stall -- same reasoning as unbooru-tagger-data's ReportOverall.
            var eta = "";
            if (rate > 0 && processed < totalSteps)
            {
                var remaining = TimeSpan.FromSeconds((totalSteps - processed) / rate);
                eta = $", ETA {remaining:hh\\:mm\\:ss}";
            }

            overallTask.Description = $"Overall ({rate:F2} steps/s{eta})";
        }

        // Not `using` -- must outlive AddTasks itself; stopped via the Dispose delegate
        // on the returned reporter once Run's progress block ends.
        var ticker = new Timer(_ => RefreshOverallDescription(), null, OverallRefreshInterval, OverallRefreshInterval);

        void ReportStepComplete()
        {
            overallTask.Increment(1);
            Volatile.Write(ref latestProcessed, (int)overallTask.Value);
            RefreshOverallDescription();
        }

        void ReportPhase(string phase)
        {
            phaseTask.IsIndeterminate = true;
            var bounded = phase.Length > DescriptionWidth ? phase[..DescriptionWidth] + "…" : phase;
            phaseTask.Description = Markup.Escape(bounded);
        }

        void ReportPhaseProgress(string phase, int completed, int total)
        {
            phaseTask.IsIndeterminate = false;
            phaseTask.MaxValue = total;
            phaseTask.Value = completed;
            var bounded = phase.Length > DescriptionWidth ? phase[..DescriptionWidth] + "…" : phase;
            phaseTask.Description = Markup.Escape(bounded);
        }

        void StopPhase() => phaseTask.StopTask();

        return new TrainingProgressReporter(ReportPhase, ReportPhaseProgress, ReportStepComplete, StopPhase, ticker.Dispose);
    }
}

/// <summary>Handles for <see cref="TrainingProgressColumns.AddTasks"/>'s two rows.</summary>
internal sealed record TrainingProgressReporter(
    Action<string> ReportPhase,
    Action<string, int, int> ReportPhaseProgress,
    Action ReportStepComplete,
    Action StopPhase,
    Action Dispose);
