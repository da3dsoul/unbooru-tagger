using Spectre.Console;
using Spectre.Console.Rendering;

namespace UnbooruTagger.Data;

/// <summary>
/// A description column with a fixed render width instead of Spectre's default of
/// sizing to whatever's currently the longest Description across all rows. That default
/// recomputes every frame, so if descriptions vary in length — a short "Overall" label
/// next to a long "Page N: fetching M image blobs..." phase, or one row's own text
/// changing length as it moves between phases — the column's width changes frame to
/// frame and the whole bar/percentage/ETA block visibly shifts left and right. Padding
/// the Description text with spaces does not fix this: Spectre measures content width
/// ignoring trailing whitespace, so a padded-but-otherwise-default column still resizes
/// to the same wrong value. Overriding <see cref="GetColumnWidth"/> is the supported way
/// to pin it — content longer than the fixed width is ellipsized rather than wrapped, so
/// an occasional over-long message (e.g. a truncated SQL exception) degrades gracefully
/// instead of corrupting the layout.
/// </summary>
internal sealed class FixedWidthDescriptionColumn(int width) : ProgressColumn
{
    public override IRenderable Render(RenderOptions options, ProgressTask task, TimeSpan deltaTime) =>
        new Markup(task.Description ?? string.Empty).Overflow(Overflow.Ellipsis);

    public override int? GetColumnWidth(RenderOptions options) => width;
}

/// <summary>
/// Shared Spectre.Console progress-bar wiring for the data pipeline's long-running
/// commands (build-small-dataset, build-large-cache), so both render the same
/// percent/rate/ETA bar instead of raw line-per-N-images console spam.
/// </summary>
internal static class ProgressBarColumns
{
    /// <summary>
    /// Width for <see cref="FixedWidthDescriptionColumn"/>. Comfortably fits every phase
    /// message this project generates in normal operation — the longest, the parallel
    /// blob-fetch phase with a worst-case page number/count, comes in under 80 — with a
    /// little headroom; only the rare truncated SQL-exception phase text is long enough
    /// to actually get ellipsized.
    /// </summary>
    private const int DescriptionWidth = 80;

    public static ProgressColumn[] Default { get; } =
    [
        new FixedWidthDescriptionColumn(DescriptionWidth),
        new ProgressBarColumn(),
        new PercentageColumn(),
        new RemainingTimeColumn(),
        new SpinnerColumn()
    ];

    /// <summary>
    /// Columns for <see cref="AddLargeCacheTasks"/>'s two rows — no
    /// <see cref="RemainingTimeColumn"/>. Spectre's ETA is a per-task velocity estimate,
    /// and neither task has a velocity that means anything: the phase row only carries a
    /// fraction while a chunked blob fetch or a page write is in flight (seconds-scale,
    /// over almost immediately) or sits indeterminate otherwise. Overall's rate
    /// genuinely is meaningful, but this pipeline's progress is inherently bursty — long
    /// stalls while a page is fetched, then a burst of writes once it lands — and
    /// Spectre's velocity window reacts to that burstiness rather than smoothing it,
    /// which is what "resets and is wildly inaccurate" was: right after a write burst
    /// the estimated speed spikes, then decays through the next stall. ReportOverall
    /// below computes its own ETA from the average rate since the run started instead,
    /// which one burst can't swing, and folds it into the Description text so no ETA
    /// column is needed at all.
    /// </summary>
    public static ProgressColumn[] LargeCacheColumns { get; } =
    [
        new FixedWidthDescriptionColumn(DescriptionWidth),
        new ProgressBarColumn(),
        new PercentageColumn(),
        new SpinnerColumn()
    ];

    /// <summary>Adds a task to <paramref name="ctx"/> and returns an <see cref="IProgress{T}"/> that keeps its bar, percentage, and description (with a live images/sec rate) in sync.</summary>
    public static IProgress<ImageBuildProgress> AddTask(ProgressContext ctx, string description)
    {
        var task = ctx.AddTask(description);
        var startedAt = DateTime.UtcNow;

        return new Progress<ImageBuildProgress>(p =>
        {
            task.MaxValue = p.Total;
            task.Value = p.Processed;

            var elapsed = (DateTime.UtcNow - startedAt).TotalSeconds;
            var rate = elapsed > 0 ? p.Processed / elapsed : 0;
            task.Description = $"{description} ({rate:F1} img/s)";
        });
    }

    /// <summary>How often the Overall row's rate/ETA text refreshes on its own, independent of when an image finishes.</summary>
    private static readonly TimeSpan OverallRefreshInterval = TimeSpan.FromSeconds(1);

    /// <summary>
    /// Adds two rows for <see cref="LargeDatasetPreprocessor.BuildAsync"/>: a phase row
    /// (what's happening right now — init steps, or the current page's step, with a
    /// real fraction while a blob fetch or a page write is in flight, since both are
    /// broken into concrete countable units), and overall corpus progress. A flat
    /// percentage alone is a poor "is this actually moving" signal once the corpus is in
    /// the millions of images, which the phase row's fraction addresses directly; a
    /// separate "current page" row on top of that added nothing, since it only ever
    /// moved during the exact same write step the phase row's own fraction now covers.
    /// </summary>
    public static LargeCacheProgressReporter AddLargeCacheTasks(ProgressContext ctx)
    {
        var phaseTask = ctx.AddTask("Initializing...");
        phaseTask.IsIndeterminate = true;

        var overallTask = ctx.AddTask("Overall");

        var startedAt = DateTime.UtcNow;
        var latestProcessed = 0;
        var latestTotal = 0;

        void RefreshOverallDescription()
        {
            var elapsed = (DateTime.UtcNow - startedAt).TotalSeconds;
            var processed = Volatile.Read(ref latestProcessed);
            var total = Volatile.Read(ref latestTotal);
            var rate = elapsed > 0 ? processed / elapsed : 0;

            // Average rate since the run started, not a short recent-velocity window:
            // this pipeline alternates long fetch stalls with fast write bursts, and a
            // window narrow enough to react quickly to real slowdowns is exactly narrow
            // enough to spike right after every burst and decay through every stall.
            // Averaging over the whole run so far can't be swung by any single page —
            // and because elapsed keeps growing every tick even while processed sits
            // still through a stall, the rate visibly ticks down during one, same as it
            // would for any other time genuinely spent on non-image work.
            var eta = "";
            if (rate > 0 && processed > 0 && processed < total)
            {
                var remaining = TimeSpan.FromSeconds((total - processed) / rate);
                eta = $", ETA {remaining:hh\\:mm\\:ss}";
            }

            overallTask.Description = $"Overall ({rate:F1} img/s{eta})";
        }

        // Refreshes the rate/ETA text on a clock instead of only when ReportOverall is
        // called: images complete in bursts (fast writes) separated by long stalls
        // (slow fetches), so an event-driven-only refresh freezes the display at
        // whatever it last showed for the entire stall, then jumps — which reads as
        // "did this hang?" and hides that the stall itself is part of what's dragging
        // overall throughput down.
        // Not `using` — this must outlive AddLargeCacheTasks itself; it's stopped via
        // the Dispose delegate on the returned reporter, once BuildAsync's run ends.
        var ticker = new Timer(_ => RefreshOverallDescription(), null, OverallRefreshInterval, OverallRefreshInterval);

        void ReportPhase(string phase)
        {
            // A fresh phase label has no known fraction of its own (the id/tag lookups
            // and the checkpoint flush are each one all-or-nothing step) until/unless a
            // ReportPhaseProgress call says otherwise, so drop back to a spinner here.
            phaseTask.IsIndeterminate = true;

            // Truncated because this text can carry caller-supplied content (a SQL
            // exception message, in practice) whose length isn't under this renderer's
            // control — FixedWidthDescriptionColumn ellipsizes overflow on its own, but
            // an unbounded string is still needlessly expensive to keep passing around.
            var bounded = phase.Length > DescriptionWidth ? phase[..DescriptionWidth] + "…" : phase;
            phaseTask.Description = Markup.Escape(bounded);
        }

        void ReportPhaseProgress(int completed, int total)
        {
            phaseTask.IsIndeterminate = false;
            phaseTask.MaxValue = total;
            phaseTask.Value = completed;
        }

        void ReportOverall(int processed, int total)
        {
            overallTask.MaxValue = total;
            overallTask.Value = processed;
            Volatile.Write(ref latestProcessed, processed);
            Volatile.Write(ref latestTotal, total);

            // Also refresh immediately so the rate/ETA text updates right when an image
            // completes rather than lagging up to a second behind the bar itself.
            RefreshOverallDescription();
        }

        return new LargeCacheProgressReporter(ReportPhase, ReportOverall, ReportPhaseProgress, ticker.Dispose);
    }
}
