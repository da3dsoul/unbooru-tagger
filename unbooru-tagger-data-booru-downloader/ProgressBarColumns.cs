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
/// One site's own progress rows — phase text (with a real fraction while a page's posts
/// are in flight), this site's current tag (its own count against a realistic per-tag
/// ceiling), and how many of the eligible tags this site's own worker has finished this
/// phase. Each configured site (danbooru, gelbooru, ...) gets one of these, since
/// <see cref="DatasetCrawler"/> now runs a dedicated concurrent worker per site instead
/// of picking one site at a time to fetch from — a single shared "current tag" row
/// stopped meaning anything once two sites could be on two different tags at once.
/// </summary>
public sealed record SiteProgressReporter(
    Action<string> ReportPhase,
    Action<int, int> ReportPhaseProgress,
    Action<string, long, long> ReportTagProgress,
    Action<long, long> ReportTagsCompleted);

/// <summary>
/// Reports what a crawl run is doing right now: one <see cref="SiteProgressReporter"/>
/// per configured site (each with its own phase/current-tag/tags-completed rows,
/// updated concurrently from that site's own worker), plus one shared
/// images-appended-this-session counter/rate row (<see cref="ReportOverall"/>) — the one
/// metric that's naturally global rather than owned by a single site.
///
/// Deliberately no shared "images appended / pre-dedup estimate" percentage — that
/// estimate sums <c>min(--max-images, count)</c> across every eligible tag, which
/// double-, triple-, ..., N-counts any image carrying N eligible tags. On a real,
/// densely-tagged corpus that inflates the denominator by orders of magnitude (a real
/// run once eligible-tag-surveyed at 34k tags priced the upper bound at ~30M image
/// slots against an actual ~12k-image corpus), so the percentage reads as permanently
/// stuck at 0% — technically not wrong, but useless as a "how far along am I" signal.
/// <see cref="ReportOverall"/>'s rate also used to divide the *lifetime* cumulative
/// image count by *this session's* elapsed time, which is wrong the instant a run is
/// resumed against an existing corpus — tracked against a session-start baseline instead.
/// </summary>
public sealed record CrawlProgressReporter(
    IReadOnlyDictionary<string, SiteProgressReporter> Sites,
    Action<long, long> ReportOverall,
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

    /// <summary>How often the images-appended row's rate text refreshes on its own, independent of when an image finishes.</summary>
    private static readonly TimeSpan OverallRefreshInterval = TimeSpan.FromSeconds(1);

    private static SiteProgressReporter AddSiteTasks(ProgressContext ctx, string site)
    {
        // AddTask's description is markup, not a literal string — an unescaped site
        // name containing/wrapped in "[...]" (which every one of these initial labels
        // is, by construction) gets parsed as a color/style tag and throws
        // InvalidOperationException the moment it's first rendered. Every other
        // Description assignment below already goes through Markup.Escape; these
        // initial AddTask calls need the same treatment.
        var phaseTask = ctx.AddTask(Markup.Escape($"[{site}] Initializing..."));
        phaseTask.IsIndeterminate = true;

        var tagProgressTask = ctx.AddTask(Markup.Escape($"[{site}] Current tag"));
        var tagsTask = ctx.AddTask(Markup.Escape($"[{site}] Tags completed"));

        void ReportPhase(string phase)
        {
            // A fresh phase label has no fraction of its own until a ReportPhaseProgress
            // call says otherwise (e.g. a page's post-by-post processing), so default to
            // a spinner rather than a stale bar from whatever the previous phase left it at.
            phaseTask.IsIndeterminate = true;
            var bounded = phase.Length > DescriptionWidth ? phase[..DescriptionWidth] + "…" : phase;
            phaseTask.Description = Markup.Escape($"[{site}] {bounded}");
        }

        void ReportPhaseProgress(int completed, int total)
        {
            phaseTask.IsIndeterminate = false;
            phaseTask.MaxValue = total;
            phaseTask.Value = completed;
        }

        void ReportTagProgress(string tagName, long completed, long target)
        {
            tagProgressTask.Description = Markup.Escape($"[{site}] Current tag: {tagName} ({completed}/{target})");
            tagProgressTask.MaxValue = target;
            // A tag can already be over target the moment its phase starts (e.g. the
            // negative top-up's "current negatives" is corpus-wide arithmetic, not
            // per-tag-owned, so it can already exceed negativeTarget on tag one) —
            // clamp so the bar reads "done", not garbage past 100%.
            tagProgressTask.Value = Math.Min(completed, target);
        }

        void ReportTagsCompleted(long completed, long total)
        {
            tagsTask.Description = Markup.Escape($"[{site}] Tags completed ({completed}/{total})");
            tagsTask.MaxValue = total;
            tagsTask.Value = completed;
        }

        return new SiteProgressReporter(ReportPhase, ReportPhaseProgress, ReportTagProgress, ReportTagsCompleted);
    }

    /// <summary><paramref name="sites"/> gets one row group each (see <see cref="AddSiteTasks"/>), rendered in the order given — plus one shared images-appended row.</summary>
    public static CrawlProgressReporter AddCrawlTasks(ProgressContext ctx, IReadOnlyList<string> sites)
    {
        var siteReporters = sites.ToDictionary(site => site, site => AddSiteTasks(ctx, site), StringComparer.Ordinal);

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
            // start absurdly inflated on a resumed run against an existing corpus and
            // only slowly decay back toward reality.
            var rate = elapsed > 0 ? sessionAppended / elapsed : 0;
            imagesTask.Description = $"Images appended (this session): {sessionAppended} ({rate:F1} img/s) — {appended} total in corpus";
        }

        // Refreshes on a clock instead of only when ReportOverall is called: a slow page
        // fetch or a long download can leave the display showing the same numbers for a
        // while otherwise, which reads as "did this hang?" rather than showing that the
        // stall itself is part of what's dragging throughput down.
        // Not `using` — stopped via the Dispose delegate once RunAsync's run ends.
        var ticker = new Timer(_ => RefreshImagesDescription(), null, OverallRefreshInterval, OverallRefreshInterval);

        void ReportOverall(long appended, long _)
        {
            sessionBaselineAppended ??= appended;
            Volatile.Write(ref latestAppended, appended);

            // Also refresh immediately so the rate text updates right when an image
            // completes rather than lagging up to a second behind the counter itself.
            RefreshImagesDescription();
        }

        return new CrawlProgressReporter(siteReporters, ReportOverall, ticker.Dispose);
    }

    /// <summary>Same per-site row shape as <see cref="AddCrawlTasks"/>, labeled for <see cref="TagRefresher"/> — its "images appended" counter would be actively misleading here since a refresh pass never appends anything, only reconciles existing rows' tags, so the shared row instead just counts sources checked.</summary>
    public static CrawlProgressReporter AddRefreshTasks(ProgressContext ctx, IReadOnlyList<string> sites)
    {
        var siteReporters = sites.ToDictionary(site => site, site => AddSiteTasks(ctx, site), StringComparer.Ordinal);

        var overallTask = ctx.AddTask("Overall (sources checked)");
        overallTask.IsIndeterminate = true;

        void ReportOverall(long checkedCount, long _)
        {
            overallTask.Description = $"Overall (sources checked) ({checkedCount})";
        }

        return new CrawlProgressReporter(siteReporters, ReportOverall, () => { });
    }
}
