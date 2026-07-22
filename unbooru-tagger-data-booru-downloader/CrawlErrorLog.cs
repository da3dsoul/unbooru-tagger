namespace UnbooruTagger.Crawler;

/// <summary>
/// Appends timestamped site-failure records to a plain-text log file in the dataset
/// directory, so a failure that happens hours into an unattended overnight run leaves a
/// durable trail even after the terminal session that showed it live is gone — the live
/// progress row only ever shows the *current* failure, not a history of every one that's
/// happened so far. One line per failure: UTC timestamp, site, and the message that
/// triggered the retry-with-backoff (see <see cref="DatasetCrawler"/>).
///
/// Plain synchronous <c>lock</c>, not an async one: failures are rare (this is the tier
/// above <see cref="TransientHttpRetry"/>'s own much more frequent short-term retries),
/// so a brief, occasional blocking file append is not worth the complexity of an async
/// primitive here.
/// </summary>
public sealed class CrawlErrorLog(string path)
{
    private readonly object _lock = new();

    public static CrawlErrorLog ForDirectory(string outputDirectory) =>
        new(System.IO.Path.Combine(outputDirectory, "crawl-errors.log"));

    public string LogPath { get; } = path;

    public void Log(string site, string message)
    {
        var line = $"{DateTimeOffset.UtcNow:O} [{site}] {message}";
        lock (_lock)
        {
            File.AppendAllLines(LogPath, [line]);
        }
    }
}
