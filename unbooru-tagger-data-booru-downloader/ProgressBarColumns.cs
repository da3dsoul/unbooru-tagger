using Spectre.Console;
using Spectre.Console.Rendering;

namespace UnbooruTagger.Crawler;

/// <summary>Same fixed-width description trick as the data pipeline's own column of this name — pins the bar/percentage block in place instead of it shifting as phase text length changes frame to frame.</summary>
internal sealed class FixedWidthDescriptionColumn(int width) : ProgressColumn
{
    public override IRenderable Render(RenderOptions options, ProgressTask task, TimeSpan deltaTime) =>
        new Markup(task.Description ?? string.Empty).Overflow(Overflow.Ellipsis);

    public override int? GetColumnWidth(RenderOptions options) => width;
}

/// <summary>
/// Reports what <see cref="DatasetCrawler"/> is doing right now: a phase-text row
/// (tag/site/checkpoint transitions, with a real fraction while processing a page's
/// posts is in flight — the one sub-step within a phase that's concretely countable),
/// plus overall appended-image progress against the pre-dedup estimate. Mirrors
/// <c>unbooru-tagger-data</c>'s <c>LargeCacheProgressReporter</c> shape/rationale: a flat
/// percentage alone is a poor "is this actually moving" signal, and a live-ticking
/// rate/ETA (independent of exactly when an image finishes) is what keeps the display
/// from looking frozen through a slow page fetch.
/// </summary>
public sealed record CrawlProgressReporter(
    Action<string> ReportPhase,
    Action<long, long> ReportOverall,
    Action<int, int> ReportPhaseProgress,
    Action Dispose);

public static class ProgressBarColumns
{
    private const int DescriptionWidth = 90;

    public static ProgressColumn[] Default { get; } =
    [
        new FixedWidthDescriptionColumn(DescriptionWidth),
        new ProgressBarColumn(),
        new PercentageColumn(),
        new SpinnerColumn()
    ];

    /// <summary>How often the Overall row's rate/ETA text refreshes on its own, independent of when an image finishes.</summary>
    private static readonly TimeSpan OverallRefreshInterval = TimeSpan.FromSeconds(1);

    public static CrawlProgressReporter AddCrawlTasks(ProgressContext ctx)
    {
        var phaseTask = ctx.AddTask("Initializing...");
        phaseTask.IsIndeterminate = true;

        var overallTask = ctx.AddTask("Overall (images appended)");

        var startedAt = DateTime.UtcNow;
        var latestAppended = 0L;
        var latestTotal = 0L;

        void RefreshOverallDescription()
        {
            var elapsed = (DateTime.UtcNow - startedAt).TotalSeconds;
            var appended = Volatile.Read(ref latestAppended);
            var total = Volatile.Read(ref latestTotal);
            var rate = elapsed > 0 ? appended / elapsed : 0;

            // Average rate since the run started, not a short recent-velocity window —
            // same rationale as the data pipeline's own Overall row: this crawl
            // alternates network-bound stalls (page fetches, downloads) with fast
            // dedup-skip bursts, and a narrow window would spike right after a burst and
            // decay through the next stall instead of giving a stable read.
            var eta = "";
            if (rate > 0 && appended > 0 && appended < total)
            {
                var remaining = TimeSpan.FromSeconds((total - appended) / rate);
                eta = $", ETA {remaining:hh\\:mm\\:ss}";
            }

            overallTask.Description = $"Overall (images appended) ({rate:F1} img/s{eta})";
        }

        // Refreshes on a clock instead of only when ReportOverall is called: a slow page
        // fetch or a long download can leave the display showing the same numbers for a
        // while otherwise, which reads as "did this hang?" rather than showing that the
        // stall itself is part of what's dragging throughput down.
        // Not `using` — stopped via the Dispose delegate once RunAsync's run ends.
        var ticker = new Timer(_ => RefreshOverallDescription(), null, OverallRefreshInterval, OverallRefreshInterval);

        void ReportPhase(string phase)
        {
            // A fresh phase label has no fraction of its own until a ReportPhaseProgress
            // call says otherwise (e.g. a page's post-by-post processing), so default to
            // a spinner rather than a stale bar from whatever the previous phase left it at.
            phaseTask.IsIndeterminate = true;
            var bounded = phase.Length > DescriptionWidth ? phase[..DescriptionWidth] + "…" : phase;
            phaseTask.Description = Markup.Escape(bounded);
        }

        void ReportPhaseProgress(int completed, int total)
        {
            phaseTask.IsIndeterminate = false;
            phaseTask.MaxValue = total;
            phaseTask.Value = completed;
        }

        void ReportOverall(long appended, long estimatedTotal)
        {
            overallTask.MaxValue = Math.Max(estimatedTotal, appended);
            overallTask.Value = appended;
            Volatile.Write(ref latestAppended, appended);
            Volatile.Write(ref latestTotal, Math.Max(estimatedTotal, appended));

            // Also refresh immediately so the rate/ETA text updates right when an image
            // completes rather than lagging up to a second behind the bar itself.
            RefreshOverallDescription();
        }

        return new CrawlProgressReporter(ReportPhase, ReportOverall, ReportPhaseProgress, ticker.Dispose);
    }
}
