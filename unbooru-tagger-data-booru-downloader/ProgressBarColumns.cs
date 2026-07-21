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
/// Reports what <see cref="DatasetCrawler"/> is doing right now, top to bottom: a
/// phase-text row (tag/site/checkpoint transitions, with a real fraction while
/// processing a page's posts is in flight), a current-tag row (this specific tag's own
/// count against a realistic per-tag ceiling — <see cref="ReportTagProgress"/>), a
/// tags-completed row (how many of the eligible tags this phase has finished —
/// <see cref="ReportTagsCompleted"/>), and an images-appended-this-session counter/rate
/// row (<see cref="ReportOverall"/>).
///
/// Deliberately NOT one combined "images appended / pre-dedup estimate" percentage —
/// that estimate sums <c>min(--max-images, count)</c> across every eligible tag, which
/// double-, triple-, ..., N-counts any image carrying N eligible tags. On a real,
/// densely-tagged corpus that inflates the denominator by orders of magnitude (a real
/// run once eligible-tag-surveyed at 34k tags priced the upper bound at ~30M image
/// slots against an actual ~12k-image corpus), so the percentage reads as permanently
/// stuck at 0% — technically not wrong, but useless as a "how far along am I" signal.
/// <see cref="ReportOverall"/>'s rate/ETA also used to divide the *lifetime* cumulative
/// image count by *this session's* elapsed time, which is wrong the instant a run is
/// resumed against an existing corpus (a huge numerator over a tiny denominator right
/// at startup) — both call sites below now track a session-start baseline instead.
/// </summary>
public sealed record CrawlProgressReporter(
    Action<string> ReportPhase,
    Action<long, long> ReportOverall,
    Action<int, int> ReportPhaseProgress,
    Action<string, long, long> ReportTagsCompleted,
    Action<string, long, long> ReportTagProgress,
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

        var tagProgressTask = ctx.AddTask("Current tag");
        var tagsTask = ctx.AddTask("Tags completed");

        // Indeterminate: there's no meaningful total to bound this against (how many
        // NEW images this session will add is exactly what heavy dedup makes
        // unpredictable) — a plain running counter + rate, not a percentage.
        var imagesTask = ctx.AddTask("Images appended (this session)");
        imagesTask.IsIndeterminate = true;

        var startedAt = DateTime.UtcNow;
        long? sessionBaselineAppended = null;
        var latestAppended = 0L;

        void RefreshImagesDescription()
        {
            var elapsed = (DateTime.UtcNow - startedAt).TotalSeconds;
            var appended = Volatile.Read(ref latestAppended);
            var sessionAppended = appended - (sessionBaselineAppended ?? appended);

            // Average rate since THIS session started, over just what THIS session
            // appended — not the lifetime cumulative count, which used to make the rate
            // (and any ETA derived from it) start absurdly inflated on a resumed run
            // against an existing corpus and only slowly decay back toward reality.
            var rate = elapsed > 0 ? sessionAppended / elapsed : 0;
            imagesTask.Description = $"Images appended (this session): {sessionAppended} ({rate:F1} img/s) — {appended} total in corpus";
        }

        // Refreshes on a clock instead of only when ReportOverall is called: a slow page
        // fetch or a long download can leave the display showing the same numbers for a
        // while otherwise, which reads as "did this hang?" rather than showing that the
        // stall itself is part of what's dragging throughput down.
        // Not `using` — stopped via the Dispose delegate once RunAsync's run ends.
        var ticker = new Timer(_ => RefreshImagesDescription(), null, OverallRefreshInterval, OverallRefreshInterval);

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

        void ReportOverall(long appended, long _)
        {
            sessionBaselineAppended ??= appended;
            Volatile.Write(ref latestAppended, appended);

            // Also refresh immediately so the rate text updates right when an image
            // completes rather than lagging up to a second behind the counter itself.
            RefreshImagesDescription();
        }

        void ReportTagsCompleted(string phaseLabel, long completed, long total)
        {
            tagsTask.Description = Markup.Escape($"Tags completed — {phaseLabel} ({completed}/{total})");
            tagsTask.MaxValue = total;
            tagsTask.Value = completed;
        }

        void ReportTagProgress(string tagName, long completed, long target)
        {
            tagProgressTask.Description = Markup.Escape($"Current tag: {tagName} ({completed}/{target})");
            tagProgressTask.MaxValue = target;
            // A tag can already be over target the moment its phase starts (e.g. the
            // negative top-up's "current negatives" is corpus-wide arithmetic, not
            // per-tag-owned, so it can already exceed negativeTarget on tag one) —
            // clamp so the bar reads "done", not garbage past 100%.
            tagProgressTask.Value = Math.Min(completed, target);
        }

        return new CrawlProgressReporter(ReportPhase, ReportOverall, ReportPhaseProgress, ReportTagsCompleted, ReportTagProgress, ticker.Dispose);
    }

    /// <summary>
    /// Same shape as <see cref="AddCrawlTasks"/> but labeled for <see cref="TagRefresher"/>:
    /// an "images appended" counter would be actively misleading here since a refresh
    /// pass never appends anything, only reconciles existing rows' tags. There's no
    /// pre-known total to bound the overall row against (unlike the crawl's pre-dedup
    /// estimate) since it's a live, resumable count of sources checked this run, so it
    /// stays indeterminate-styled — a plain running counter, not a percentage.
    /// </summary>
    public static CrawlProgressReporter AddRefreshTasks(ProgressContext ctx)
    {
        var phaseTask = ctx.AddTask("Initializing...");
        phaseTask.IsIndeterminate = true;

        var overallTask = ctx.AddTask("Overall (sources checked)");
        overallTask.IsIndeterminate = true;

        void ReportPhase(string phase)
        {
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

        void ReportOverall(long checkedCount, long _)
        {
            overallTask.Description = $"Overall (sources checked) ({checkedCount})";
        }

        // refresh-tags has no "tags" concept to bound a percentage against — sources are
        // resumable but not counted up front, so these rows stay unused.
        return new CrawlProgressReporter(ReportPhase, ReportOverall, ReportPhaseProgress, (_, _, _) => { }, (_, _, _) => { }, () => { });
    }
}
