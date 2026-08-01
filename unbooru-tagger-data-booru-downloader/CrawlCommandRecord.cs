using System.Text.Json;

namespace UnbooruTagger.Crawler;

/// <summary>
/// Persists the exact options a <c>crawl</c> run was invoked with to <c>--output-dir</c>,
/// so a later re-run can pass <c>--resume</c> instead of the caller needing to remember (or
/// dig back out of shell history) <c>--min-images</c>/<c>--max-images</c>/<c>--input-size</c>/
/// API credentials/etc. This is separate from — and doesn't replace — <c>crawl</c>'s own
/// data-level resumability (<c>crawl.sqlite</c>'s per-tag/site cursors,
/// <c>images.bin.resume</c>); those already let a re-run pick back up mid-corpus as long as
/// you can reconstruct the original command. This only saves the invocation itself from
/// being lost.
/// </summary>
public sealed record CrawlCommandRecord(
    string[] Sites,
    int MinImages,
    int MaxImages,
    int InputSize,
    string? DanbooruLogin,
    string? DanbooruApiKey,
    string? GelbooruApiKey,
    string? GelbooruUserId,
    double RateDanbooru,
    double RateGelbooru,
    int NegativeTarget,
    int VocabCompactInterval,
    double NegativeCooccurrenceRatio,
    int NegativeCooccurrenceMinExamples,
    int MaxHardNegativeSources)
{
    public const string FileName = "last-crawl-command.json";

    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

    public static async Task SaveAsync(string outputDirectory, CrawlCommandRecord record, CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(outputDirectory);
        var path = Path.Combine(outputDirectory, FileName);
        await File.WriteAllTextAsync(path, JsonSerializer.Serialize(record, JsonOptions), cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Reads the record <see cref="SaveAsync"/> last wrote — <see langword="null"/> if it doesn't exist yet.</summary>
    public static async Task<CrawlCommandRecord?> TryLoadAsync(string outputDirectory, CancellationToken cancellationToken = default)
    {
        var path = Path.Combine(outputDirectory, FileName);
        if (!File.Exists(path))
            return null;

        var json = await File.ReadAllTextAsync(path, cancellationToken).ConfigureAwait(false);
        return JsonSerializer.Deserialize<CrawlCommandRecord>(json, JsonOptions);
    }
}
