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
/// Reports what <see cref="DatasetCrawler"/> is doing right now via a phase-text row
/// (tag/site/phase transitions — "Starting tag '1girl' (12/438 eligible, target 1000,
/// site: gelbooru)"), plus overall appended-image progress against the pre-dedup
/// estimate. Mirrors <c>unbooru-tagger-data</c>'s <c>LargeCacheProgressReporter</c>
/// shape/rationale: a flat percentage alone is a poor "is this actually moving" signal,
/// and encoding transitions as the phase row's text (rather than separate scrolling
/// console lines) is what plays nicely with an active Spectre <c>Progress</c> region.
/// </summary>
public sealed record CrawlProgressReporter(
    Action<string> ReportPhase,
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

    public static CrawlProgressReporter AddCrawlTasks(ProgressContext ctx)
    {
        var phaseTask = ctx.AddTask("Initializing...");
        phaseTask.IsIndeterminate = true;

        var overallTask = ctx.AddTask("Overall (images appended)");

        void ReportPhase(string phase)
        {
            phaseTask.IsIndeterminate = false;
            var bounded = phase.Length > DescriptionWidth ? phase[..DescriptionWidth] + "…" : phase;
            phaseTask.Description = Markup.Escape(bounded);
            phaseTask.Value = phaseTask.MaxValue; // always shown as "complete" — this row communicates via text, not fraction
        }

        void ReportOverall(long appended, long estimatedTotal)
        {
            overallTask.MaxValue = Math.Max(estimatedTotal, appended);
            overallTask.Value = appended;
        }

        return new CrawlProgressReporter(ReportPhase, ReportOverall, () => { });
    }
}
