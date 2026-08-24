using System.Net;
using System.Security.Cryptography;
using SkiaSharp;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Crawler;

/// <summary>
/// The full, deterministic target ordering a fresh rebuild of the pixel cache must
/// reproduce to line up with an ALREADY-COMPACTED cache's existing rows -- i.e. the
/// exact same (ascending old <c>CacheRowIndex</c>, excluding sole-unsupported-media-source
/// images) sequence <see cref="Program"/>'s <c>prune-unsupported-media</c> command fed to
/// <c>PreprocessedDatasetCacheCompactor.DropRows</c> in the first place. Recomputed
/// entirely from <c>crawl.sqlite</c>, which was never touched by the compaction that
/// went wrong (its own crawl.sqlite-side update rolled back when the run was killed
/// before committing) -- so this reproduces the SAME plan byte-for-byte on every call,
/// which is exactly what lets index <c>i</c> here mean the same thing as index
/// <c>i</c> in the damaged cache still sitting on disk.
/// </summary>
public sealed record DatasetRecoveryPlan(
    IReadOnlyList<string> OrderedMd5s,
    IReadOnlySet<string> SoleSourceMd5s,
    IReadOnlyList<(string Md5, string Site, long PostId)> ExtraSourcesToStrip);

/// <param name="WrittenMd5s">Every Md5 actually appended to the staging cache, in final row order -- shorter than the plan's own <see cref="DatasetRecoveryPlan.OrderedMd5s"/> by however many permanently failed.</param>
/// <param name="PermanentlyFailedMd5s">Every Md5 no candidate source could recover, in the order encountered.</param>
public sealed record RecoveryFillResult(IReadOnlyList<string> WrittenMd5s, IReadOnlyList<string> PermanentlyFailedMd5s);

public static class DatasetRecovery
{
    /// <summary>Md5\tReason, one per line, append-only -- both the resumable durable record of what's already been tried-and-given-up-on, and the human-readable list a caller reports at the end.</summary>
    public const string FailuresLogFileName = "recovery-failures.log";

    /// <summary>
    /// Replays <c>prune-unsupported-media</c>'s own row-selection logic (see
    /// <see cref="CrawlDatabase.GetUnsupportedMediaGroupsAsync"/>/
    /// <see cref="CrawlDatabase.GetUnsupportedMediaSourceRowsAsync"/>, the same queries
    /// that command itself uses) against <paramref name="db"/>'s CURRENT state.
    /// </summary>
    public static async Task<DatasetRecoveryPlan> ComputePlanAsync(CrawlDatabase db, CancellationToken cancellationToken = default)
    {
        var images = await db.GetAllImagesAsync(cancellationToken);
        var groups = await db.GetUnsupportedMediaGroupsAsync(cancellationToken);
        var soleSourceMd5s = groups
            .Where(g => g.UnsupportedSourceCount == g.TotalSourceCount)
            .Select(g => g.Md5)
            .ToHashSet(StringComparer.Ordinal);

        var badRows = await db.GetUnsupportedMediaSourceRowsAsync(cancellationToken);
        var extraSourcesToStrip = badRows
            .Where(r => !soleSourceMd5s.Contains(r.Md5))
            .Select(r => (r.Md5, r.Site, r.PostId))
            .ToList();

        var orderedMd5s = images
            .OrderBy(i => i.CacheRowIndex)
            .Where(i => !soleSourceMd5s.Contains(i.Md5))
            .Select(i => i.Md5)
            .ToList();

        return new DatasetRecoveryPlan(orderedMd5s, soleSourceMd5s, extraSourcesToStrip);
    }

    /// <summary>
    /// Appends every one of <paramref name="orderedMd5s"/> from
    /// <paramref name="stagingDirectory"/>'s current <c>writer.ImageCount</c> onward
    /// (i.e. everywhere <see cref="PreprocessedDatasetCacheCompactor.CopyPrefix"/>'s
    /// valid-prefix copy left off), re-downloading each from its recorded source
    /// URL(s), decoding, and re-encoding at <paramref name="inputSize"/>. Downloads run
    /// as a sliding window of up to <paramref name="concurrency"/> in flight at once
    /// (not synchronous batches -- see the loop's own comment), but pixel/tag-row
    /// writes stay in strict plan order regardless of which download actually finishes
    /// first. Resumable every <paramref name="concurrency"/> entries: both the pixel
    /// writer's own durable <c>ImageCount</c> and <see cref="FailuresLogFileName"/> are
    /// flushed together at each checkpoint, and re-running this after an interruption
    /// picks up exactly where it left off (their combined count IS the resume position
    /// -- every plan entry considered ends up in exactly one of the two, in order, so
    /// nothing needs a separate progress marker).
    ///
    /// Tag rows are pulled from <paramref name="tagRowSourceDirectory"/> (the DAMAGED
    /// cache still sitting in the original dataset directory) rather than recomputed --
    /// that corruption only ever hit pixel bytes (see the incident this class exists to
    /// recover from), never <c>tag_rows.sqlite</c>, which lives in a wholly separate
    /// file/connection untouched by whatever went wrong on the pixel side.
    ///
    /// A successfully re-downloaded file's own MD5 is checked against the identity
    /// <paramref name="orderedMd5s"/> already claims for that row before it's trusted --
    /// a site can replace or remove content at a URL that used to serve something else,
    /// and silently accepting that would reintroduce a wrong image under someone else's
    /// identity instead of just leaving a visible gap.
    /// </summary>
    public static async Task<RecoveryFillResult> FillRemainingAsync(
        string stagingDirectory,
        string tagRowSourceDirectory,
        int inputSize,
        IReadOnlyList<string> orderedMd5s,
        IReadOnlyDictionary<string, IReadOnlyList<(string Site, string PostUrl)>> sourceUrlsByMd5,
        HttpClient downloadClient,
        IReadOnlyDictionary<string, IRateLimiter> rateLimiterBySite,
        IReadOnlyDictionary<string, double> ratesBySite,
        int concurrency,
        Action<int, int>? onProgress = null,
        CancellationToken cancellationToken = default)
    {
        var siteSchedule = new SiteRoundRobin(ratesBySite);
        var failuresLogPath = Path.Combine(stagingDirectory, FailuresLogFileName);
        var alreadyFailedMd5s = File.Exists(failuresLogPath)
            ? File.ReadLines(failuresLogPath).Select(l => l.Split('\t')[0]).ToList()
            : [];
        var failedSet = alreadyFailedMd5s.ToHashSet(StringComparer.Ordinal);

        using var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(stagingDirectory, inputSize);
        var consideredCount = writer.ImageCount + alreadyFailedMd5s.Count;

        // Reconstruct the caller-facing "what actually got written, in order" list for
        // everything a prior (interrupted) run already resolved -- pure in-memory replay
        // against the already-durable failures log, no re-download needed.
        var writtenMd5s = new List<string>(orderedMd5s.Count);
        for (var i = 0; i < consideredCount; i++)
        {
            if (!failedSet.Contains(orderedMd5s[i]))
                writtenMd5s.Add(orderedMd5s[i]);
        }

        await using var failuresLog = new StreamWriter(new FileStream(failuresLogPath, FileMode.Append, FileAccess.Write));

        var total = orderedMd5s.Count;
        var index = consideredCount;
        using var tagRowEnumerator = PreprocessedDatasetCacheCompactor.StreamTagRows(tagRowSourceDirectory, total, index).GetEnumerator();

        onProgress?.Invoke(index, total);

        // A sliding window of up to `concurrency` in-flight downloads, NOT batches that
        // wait for every member to finish before starting the next group. Batching used
        // to mean one slow/retried request (a 429 backoff, a flaky connection --
        // TransientHttpRetry's job is exactly to absorb these, which takes real time)
        // stalled every other worker in its batch AND blocked the next batch from even
        // starting, even though the site rate limiters would otherwise have allowed
        // continuous dispatch. Confirmed for real: observed throughput was under half of
        // the two site rate limiters' combined theoretical maximum. Here, the moment one
        // slot's result is consumed (in strict plan order, so the pixel/tag-row writes
        // stay sequential regardless of which download actually finished first), a
        // replacement launches immediately -- only the SPECIFIC next-in-order item's
        // straggler blocks progress, not the other `concurrency - 1` slots' worth of
        // otherwise-idle capacity.
        var window = new Dictionary<int, Task<EncodedImage?>>(concurrency);
        var nextToLaunch = index;

        void LaunchNext()
        {
            if (nextToLaunch >= total)
                return;
            var idx = nextToLaunch++;
            var md5 = orderedMd5s[idx];
            var candidates = sourceUrlsByMd5.GetValueOrDefault(md5, Array.Empty<(string, string)>());
            var ordered = siteSchedule.OrderCandidates(candidates);
            window[idx] = DownloadAndEncodeAsync(md5, ordered, downloadClient, rateLimiterBySite, inputSize, cancellationToken);
        }

        for (var i = 0; i < concurrency; i++)
            LaunchNext();

        var sinceCheckpoint = 0;
        while (index < total)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var md5 = orderedMd5s[index];
            var result = await window[index].ConfigureAwait(false);
            window.Remove(index);

            if (!tagRowEnumerator.MoveNext() || tagRowEnumerator.Current.Index != index)
                throw new InvalidDataException($"'{tagRowSourceDirectory}' is missing a tag-row entry for row {index} -- refusing to continue.");

            if (result is { } encoded)
            {
                writer.Append(encoded, tagRowEnumerator.Current.TagRows);
                writtenMd5s.Add(md5);
            }
            else
            {
                await failuresLog.WriteLineAsync($"{md5}\tno working source").ConfigureAwait(false);
            }

            LaunchNext();
            index++;
            sinceCheckpoint++;

            // Same checkpoint granularity as the old batch design (one checkpoint per
            // `concurrency` items) -- this fix is about keeping the window full between
            // checkpoints, not about how often progress is made durable.
            if (sinceCheckpoint >= concurrency || index == total)
            {
                writer.Flush();
                await failuresLog.FlushAsync().ConfigureAwait(false);
                sinceCheckpoint = 0;
                onProgress?.Invoke(index, total);
            }
        }

        var allFailed = File.Exists(failuresLogPath)
            ? File.ReadLines(failuresLogPath).Select(l => l.Split('\t')[0]).ToList()
            : new List<string>();
        return new RecoveryFillResult(writtenMd5s, allFailed);
    }

    /// <summary>
    /// Decides which site to try FIRST for each image that has sources on more than
    /// one, using "smooth weighted round-robin" (the same algorithm nginx's own
    /// upstream balancer uses) proportional to each site's configured request rate --
    /// e.g. a 4:2 danbooru:gelbooru rate ratio interleaves roughly
    /// danbooru,danbooru,gelbooru,danbooru,danbooru,gelbooru,... across images that have
    /// a choice, rather than a fixed "always prefer danbooru" rule. A fixed preference
    /// was tried first and confirmed for real to cap the whole re-download phase's
    /// throughput at just the preferred site's own rate: every image with a source on
    /// the preferred site (the common case) piled onto that ONE site's rate limiter,
    /// leaving the other site's capacity sitting completely idle the entire time the
    /// preferred site had a candidate available. Every pick call happens from
    /// <see cref="FillRemainingAsync"/>'s single-threaded scheduling loop (never
    /// concurrently with another pick), so the running credit state needs no locking.
    /// </summary>
    private sealed class SiteRoundRobin(IReadOnlyDictionary<string, double> weightsBySite)
    {
        private readonly Dictionary<string, double> _credits = weightsBySite.Keys.ToDictionary(s => s, _ => 0.0, StringComparer.Ordinal);
        private readonly double _totalWeight = weightsBySite.Values.Sum();

        /// <summary>Reorders one image's candidates so whichever site the running schedule currently favors goes first -- a no-op when the image only has sources on a single site (nothing to choose between), and stable otherwise (each site's own relative order, already most-recently-observed-first, is preserved).</summary>
        public IReadOnlyList<(string Site, string PostUrl)> OrderCandidates(IReadOnlyList<(string Site, string PostUrl)> candidates)
        {
            var distinctSites = candidates.Select(c => c.Site).Distinct(StringComparer.Ordinal).ToList();
            if (distinctSites.Count <= 1)
                return candidates;

            var preferred = Pick(distinctSites);
            return candidates.OrderBy(c => c.Site == preferred ? 0 : 1).ToList();
        }

        private string Pick(IReadOnlyList<string> availableSites)
        {
            foreach (var site in _credits.Keys)
                _credits[site] += weightsBySite.GetValueOrDefault(site);

            var best = availableSites.OrderByDescending(s => _credits.GetValueOrDefault(s)).First();
            _credits[best] -= _totalWeight;
            return best;
        }
    }

    /// <summary>Tries each of <paramref name="candidates"/> in order (whichever site <see cref="SiteRoundRobin"/> picked first, most-recently-observed first within a site -- see <see cref="CrawlDatabase.GetAllImageSourceUrlsAsync"/>) until one downloads, decodes, and MD5-matches <paramref name="expectedMd5"/>; <see langword="null"/> if every candidate fails.</summary>
    private static async Task<EncodedImage?> DownloadAndEncodeAsync(
        string expectedMd5,
        IReadOnlyList<(string Site, string PostUrl)> candidates,
        HttpClient downloadClient,
        IReadOnlyDictionary<string, IRateLimiter> rateLimiterBySite,
        int inputSize,
        CancellationToken cancellationToken)
    {
        foreach (var (site, postUrl) in candidates)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (!rateLimiterBySite.TryGetValue(site, out var rateLimiter))
                continue; // no client configured for this site this run -- try the next candidate

            var fileUrl = new Uri(postUrl);
            HttpResponseMessage response;
            try
            {
                response = await TransientHttpRetry.SendWithRetryAsync(
                    async () =>
                    {
                        await rateLimiter.WaitAsync(cancellationToken).ConfigureAwait(false);
                        using var request = new HttpRequestMessage(HttpMethod.Get, fileUrl);
                        request.Headers.Referrer = new Uri($"{fileUrl.Scheme}://{fileUrl.Host}/");
                        return await downloadClient.SendAsync(request, HttpCompletionOption.ResponseContentRead, cancellationToken).ConfigureAwait(false);
                    },
                    fileUrl, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is HttpRequestException or IOException)
            {
                continue; // this source is unreachable -- try the next candidate
            }

            using (response)
            {
                if (!response.IsSuccessStatusCode) // covers 404/410 (permanently gone) same as everything else here
                    continue;

                byte[] bytes;
                try
                {
                    bytes = await response.Content.ReadAsByteArrayAsync(cancellationToken).ConfigureAwait(false);
                }
                catch (IOException)
                {
                    continue;
                }

                var actualMd5 = Convert.ToHexStringLower(MD5.HashData(bytes));
                if (!string.Equals(actualMd5, expectedMd5, StringComparison.OrdinalIgnoreCase))
                    continue; // site now serves different content at this URL -- don't trust it silently

                try
                {
                    using var bitmap = SKBitmap.Decode(bytes) ?? throw new InvalidDataException();
                    return ImagePreprocessing.Encode(bitmap, inputSize);
                }
                catch (InvalidDataException)
                {
                    continue; // corrupt/undecodable -- try the next candidate
                }
            }
        }

        return null; // every candidate source failed
    }
}
