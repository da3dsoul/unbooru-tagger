namespace UnbooruTagger.Crawler;

/// <summary>
/// File extensions this project has no business ever crawling: video containers
/// (<c>.webm</c>/<c>.mp4</c>/<c>.mov</c>/<c>.avi</c>) that SkiaSharp can't decode as a
/// still image at all, and <c>.swf</c>/<c>.gif</c> — GIF is the more insidious one of
/// the two, since SkiaSharp doesn't fail on it: it silently decodes just the first
/// frame as if it were the whole picture, so a corrupt/blank/mid-transition frame gets
/// credited as a tag's real, complete artwork instead of the download failing loudly.
///
/// Single source of truth for both <see cref="DatasetCrawler"/> (skips these before
/// ever downloading — see its own <c>IsUnsupportedMedia</c>) and the
/// <c>prune-unsupported-media</c> command (retroactively finds/removes ones that
/// already made it into an existing corpus from before that skip existed).
/// </summary>
public static class UnsupportedMedia
{
    public static readonly IReadOnlyList<string> Extensions = [".gif", ".webm", ".mp4", ".mov", ".avi", ".swf"];

    public static bool IsUnsupportedMediaUrl(string url) =>
        Extensions.Any(ext => url.EndsWith(ext, StringComparison.OrdinalIgnoreCase));

    /// <summary>Every extension as a standalone SQL <c>LIKE</c> predicate against <paramref name="column"/>, OR'd together — built from <see cref="Extensions"/> so the SQL-side check can never drift from the in-process one. Extensions are fixed internal constants, never external input, so inlining them into the SQL text is safe.</summary>
    public static string BuildSqlLikeClause(string column) =>
        string.Join(" OR ", Extensions.Select(ext => $"LOWER({column}) LIKE '%{ext}'"));
}
