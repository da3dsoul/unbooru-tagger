using System.Collections.Concurrent;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Crawler;

/// <summary>
/// A tag whose combined-across-both-sites unique image count still fell short of
/// <c>--max-images</c> once both sites' post listings were fully exhausted — the one
/// case dedup losses can't be compensated for by fetching more, because there was
/// nothing left to fetch. Returned by <see cref="DatasetCrawler.RunAsync"/> for the
/// caller to report; it's a real gap in the corpus, not just extra requests spent.
/// </summary>
public sealed record TagShortfall(string TagName, int Achieved, int Target);

/// <summary><see cref="DatasetCrawler.RunAsync"/>'s outcome. A site that failed at some point during the run never shows up here — it's retried with backoff until it succeeds or the run is cancelled (see <see cref="DatasetCrawler.RunSiteTagPhaseAsync"/>), not dropped; check <see cref="CrawlErrorLog"/> for a durable record of what happened while you weren't watching.</summary>
public sealed record CrawlResult(IReadOnlyList<TagShortfall> Shortfalls);

/// <summary>Up-front, pre-dedup estimate of what a crawl will cost — printed by <c>survey-tags</c> and recomputed as the first step of <c>crawl</c>.</summary>
public sealed record CrawlEstimate(
    int EligibleTagCount,
    long EstimatedImageSlots,
    long EstimatedRequests,
    TimeSpan EstimatedWallClockTime);

/// <summary>
/// Every piece of in-memory state a crawl run needs, live and accurate, without a DB
/// round trip per post: exact-md5 and perceptual-hash dedup indexes, each surveyed tag's
/// running combined-positive count (for quota decisions), and the newly-downloaded
/// images/sources not yet committed to <c>crawl.sqlite</c>. Seeded once from durable
/// state at the start of a run (so a resumed run picks up exactly where the last
/// successful checkpoint left off), then updated immediately as posts are processed —
/// only the <em>durable</em> copy of new images/sources lags, deliberately, until the
/// next per-page checkpoint (see <see cref="DatasetCrawler.RunAsync"/>).
///
/// One concurrent worker task runs per configured site (see <see cref="DatasetCrawler.RunAsync"/>),
/// all sharing this SAME instance — every field except <see cref="CombinedPositiveCounts"/>
/// is only ever touched while holding <c>RunAsync</c>'s <c>stateLock</c>, so a plain
/// <see cref="Dictionary{TKey,TValue}"/>/<see cref="List{T}"/> is fine for those.
/// <see cref="CombinedPositiveCounts"/> is the one field read <em>without</em> that lock
/// too — every site's own <c>shouldContinue</c>/tag-progress check reads it on every
/// loop iteration, and taking a full lock for just that read would serialize the two
/// sites' otherwise-independent hot loops — so it's a
/// <see cref="ConcurrentDictionary{TKey,TValue}"/> instead: safe for that unsynchronized
/// read to happen alongside a write elsewhere, at the cost of the read possibly seeing
/// a value from a moment ago (already true of the pre-concurrency single-threaded
/// design too — a page's quota check only ever reflected state as of its last check).
/// </summary>
internal sealed class CrawlWorkingState
{
    public required Dictionary<string, int> KnownImages { get; init; }
    public required List<(ulong Hash, string Md5)> HashIndex { get; init; }
    public required ConcurrentDictionary<string, int> CombinedPositiveCounts { get; init; }

    /// <summary>
    /// Per (site, tag), how many images <em>that site itself</em> is the reason count
    /// toward <see cref="CombinedPositiveCounts"/> — either a brand-new row it was first
    /// to find, or a merge where its copy carried a tag an earlier-seen copy didn't (see
    /// <see cref="DatasetCrawler.MergeDuplicateTags"/>). A site "catching up" to an image
    /// the other site already found (a plain additional-source record, nothing new)
    /// never increments this. This is the fairness floor's own counter — see
    /// <see cref="DatasetCrawler.RunAsync"/>'s doc comment for why
    /// <see cref="CombinedPositiveCounts"/> alone let a faster site starve a slower one
    /// out of ever discovering anything of its own. One <see cref="ConcurrentDictionary{TKey,TValue}"/>
    /// per configured site, set up once before any site worker starts, so no locking is
    /// needed to add/remove a site's own entry later.
    /// </summary>
    public required IReadOnlyDictionary<string, ConcurrentDictionary<string, int>> SitePositiveCounts { get; init; }

    /// <summary>
    /// Each already-known image's current, live tag-row-index set, keyed by its cache
    /// row — the in-memory mirror of what's durably in <c>tag_rows.jsonl</c> as of the
    /// last checkpoint, kept live so a duplicate found later in the same run can tell
    /// whether the other site's tags are actually new without a file read. Seeded once
    /// from <see cref="PreprocessedDatasetCacheWriter.ReadCommittedTagRows"/> at the
    /// start of a run.
    /// </summary>
    public required Dictionary<int, HashSet<int>> ImageTagRowsByCacheRow { get; init; }

    /// <summary>Cache rows whose tag set gained something new since the last checkpoint — the merge target for the next <see cref="PreprocessedDatasetCacheWriter.MergeTagRows"/> call.</summary>
    public HashSet<int> DirtyCacheRows { get; } = [];

    /// <summary>Flat, one-per-occurrence list of tag names a duplicate-image merge newly credited since the last checkpoint — mirrors <see cref="PendingNewImage.EligibleTags"/>'s role but for images that aren't new rows.</summary>
    public List<string> PendingMergedTagCounts { get; } = [];

    public List<PendingNewImage> PendingNewImages { get; } = [];
    public List<PendingAdditionalSource> PendingAdditionalSources { get; } = [];

    public int CombinedPositiveCount(string tagName) => CombinedPositiveCounts.GetValueOrDefault(tagName);
}

/// <summary>
/// Implements the <c>crawl</c> command: a rarest-eligible-tag-first positive pass across
/// every configured site, followed by an automatic negative top-up pass. Downloads never
/// persist as a raw-file corpus — each new (post-dedup) image goes to a <c>.tmp</c>
/// scratch file just long enough to decode/normalize via
/// <see cref="ImagePreprocessing.LoadAndNormalize(string, int)"/> and append to the same
/// <see cref="PreprocessedDatasetCacheWriter"/>/<see cref="TagVocabulary"/> format
/// <c>build-large-cache</c> produces, so <c>--output-dir</c> is immediately a trainable
/// dataset directory, not a raw dump needing a separate import step.
///
/// Each configured site runs as its own concurrent worker (see <see cref="RunSiteTagPhaseAsync"/>),
/// walking every eligible tag independently at whatever pace its own rate limit allows,
/// rather than the two sites taking turns fetching one page at a time — that used to
/// bottleneck the whole run on round-robin fairness even though the sites' rate limits
/// are already independent. Every site worker shares the same <see cref="CrawlWorkingState"/>,
/// <see cref="TagVocabulary"/>, and <see cref="PreprocessedDatasetCacheWriter"/>, guarded
/// by one <see cref="SemaphoreSlim"/> (<c>stateLock</c> in <see cref="RunAsync"/>) so a
/// post's dedup check and any commit to shared state is always serialized — the slow
/// part (the network download itself) deliberately happens outside the lock so the two
/// sites' downloads can actually overlap; see <see cref="ProcessPostAsync"/> for the
/// check-download-recheck pattern that keeps that safe against two sites discovering the
/// same image at nearly the same moment.
///
/// A site's worker never gives up: a page fetch or download that exhausts
/// <see cref="TransientHttpRetry"/>'s own short-term retries logs the failure to
/// <see cref="CrawlErrorLog"/>, shows a clear "ERROR ... retrying at HH:mm:ss" status on
/// that site's own progress row, waits (20 minutes by default), and tries the exact same
/// page again — indefinitely, since this is meant to run unattended for hours/days and a
/// temporary outage (a router reboot, a site's maintenance window) shouldn't need a human
/// to notice and restart it. Each site's positive-then-negative work runs as one
/// self-contained worker rather than the whole run barrier-syncing every site between
/// phases, specifically so one site stuck retrying can never block the other's progress —
/// a hard cross-site barrier plus a site that never gives up would otherwise deadlock the
/// entire run the moment one site had a real, lasting outage.
///
/// Checkpoints once per page (cache flush, vocabulary delta, buffered dedup/count
/// writes, then the pagination cursor — in that order) rather than every N images, the
/// same model <c>unbooru-tagger-data-unbooru-import</c>'s <c>LargeDatasetPreprocessor</c>
/// and <c>unbooru-tagger-training</c> use: a crash can cost at most one page's worth of
/// work, and never leaves the dedup index referencing cache rows the cache file doesn't
/// actually have (see <see cref="CrawlDatabase.CommitPendingImagesAsync"/> for what goes
/// wrong if the ordering is reversed).
/// </summary>
public static class DatasetCrawler
{
    private const string PositivePhase = "positive";
    private const string NegativePhase = "negative";

    /// <summary>
    /// Max <see cref="PerceptualHash.HammingDistance"/> (out of 64 bits) for two images to
    /// be treated as the same cross-site re-encode. Calibrated against measured data, not
    /// folklore: an experiment re-encoding a synthetic test image at JPEG quality 95→30
    /// measured pure recompression noise at 0-2 bits, while cropping (3-15% off an edge),
    /// swapped text overlays, and genuinely different images all measured 8+ — so 2 is the
    /// tightest value that still absorbs realistic recompression noise with a small margin,
    /// without room left for a crop/edit/different-image false positive. Note this can't be
    /// airtight either way: this hash only looks at the low-frequency 8x8 DCT block (that's
    /// what makes it survive recompression at all), so a small enough localized edit (e.g. a
    /// small added element) can leave the hash completely unchanged regardless of how tight
    /// this threshold is — a limitation of the hash itself, not something a distance
    /// cutoff can fix. A looser threshold also risks collapsing two genuinely different
    /// (but visually similar) images into one, silently dropping a real training example
    /// instead of a true duplicate — the previous default of 6 traded more of that risk for
    /// more tolerance of resize-driven noise (two sites serving the same source at
    /// meaningfully different resolutions measured as high as 8 in the same experiment);
    /// 2 accepts missing that case in exchange for tighter precision.
    /// </summary>
    private const int MaxHammingDistance = 2;

    public static CrawlEstimate Estimate(
        IReadOnlyList<TagSurveyResult> allTags,
        int minImages,
        int maxImages,
        IReadOnlyDictionary<string, int> pageSizeBySite,
        IReadOnlyDictionary<string, double> requestsPerSecondBySite,
        int tagSurveyRequestsMade)
    {
        var eligibleTags = allTags.Where(t => TagEligibility.IsEligible(t, minImages)).ToList();
        var imageSlots = TagEligibility.EstimateImageSlots(eligibleTags, minImages, maxImages);

        // Best-case: split evenly across sites, each page at that site's own hard cap.
        var averagePageSize = pageSizeBySite.Values.Count() > 0 ? pageSizeBySite.Values.Average() : 1;
        var estimatedRequests = tagSurveyRequestsMade + (long)Math.Ceiling(imageSlots / Math.Max(1, averagePageSize));

        var combinedRatePerSecond = requestsPerSecondBySite.Values.Sum();
        var estimatedSeconds = combinedRatePerSecond > 0 ? estimatedRequests / combinedRatePerSecond : 0;

        return new CrawlEstimate(eligibleTags.Count, imageSlots, estimatedRequests, TimeSpan.FromSeconds(estimatedSeconds));
    }

    public static async Task<CrawlResult> RunAsync(
        CrawlDatabase db,
        IReadOnlyDictionary<string, IBooruClient> clientsBySite,
        HttpClient downloadClient,
        string outputDirectory,
        int inputSize,
        int minImages,
        int maxImages,
        int negativeTarget,
        int vocabCompactIntervalPages,
        CrawlProgressReporter? progress,
        CancellationToken cancellationToken,
        Func<TimeSpan, CancellationToken, Task>? retryDelay = null)
    {
        retryDelay ??= Task.Delay;
        var errorLog = CrawlErrorLog.ForDirectory(outputDirectory);
        var sites = clientsBySite.Keys.ToList();

        // Setup below is shared, one-time work before any site worker starts — every
        // site's row briefly shows the same status rather than picking one arbitrarily.
        void ReportSetupPhase(string phase)
        {
            foreach (var siteReporter in progress?.Sites.Values ?? [])
                siteReporter.ReportPhase(phase);
        }

        ReportSetupPhase("Loading tag survey...");
        var allTags = await db.GetAllSurveyedTagsAsync(cancellationToken).ConfigureAwait(false);
        var eligibleTags = CrawlScheduling.RarestFirst(allTags.Where(t => TagEligibility.IsEligible(t, minImages))).ToList();
        var estimatedTotal = TagEligibility.EstimateImageSlots(eligibleTags, minImages, maxImages);

        ReportSetupPhase("Loading tag vocabulary...");
        var vocabularyPath = Path.Combine(outputDirectory, "tag_vocabulary.json");
        var vocabularyDeltaPath = Path.Combine(outputDirectory, "tag_vocabulary.delta.jsonl");
        var vocabulary = File.Exists(vocabularyPath)
            ? TagVocabulary.Load(vocabularyPath, vocabularyDeltaPath)
            : TagVocabulary.CreateEmpty();
        if (!File.Exists(vocabularyPath))
            vocabulary.Save(vocabularyPath);

        var eligibleTagSet = eligibleTags.Select(t => t.Name).ToHashSet(StringComparer.Ordinal);

        var tempDir = Path.Combine(outputDirectory, ".tmp");
        Directory.CreateDirectory(tempDir);
        foreach (var leftover in Directory.EnumerateFiles(tempDir))
            File.Delete(leftover); // safe: nothing durable is recorded until after a successful checkpoint

        ReportSetupPhase("Opening cache writer...");
        using var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(outputDirectory, inputSize);

        // Seeded once from durable state (accurate as of the last successful checkpoint,
        // exactly in step with the cache file writer above thanks to its own
        // truncate-to-last-flush resume logic), then kept live in memory from here on —
        // see CrawlWorkingState's own doc comment for why the durable copies deliberately
        // lag behind these during a run.
        ReportSetupPhase("Loading dedup index and tag counters...");
        var existingImages = await db.GetAllImagesAsync(cancellationToken).ConfigureAwait(false);
        CacheConsistency.Validate(existingImages, writer.ImageCount, outputDirectory);
        var combinedPositiveCounts = await db.GetAllCombinedPositiveCountsAsync(cancellationToken).ConfigureAwait(false);
        var committedTagRows = writer.ReadCommittedTagRows();
        var state = new CrawlWorkingState
        {
            KnownImages = existingImages.ToDictionary(e => e.Md5, e => e.CacheRowIndex, StringComparer.Ordinal),
            HashIndex = existingImages.Select(e => (e.PHash, e.Md5)).ToList(),
            CombinedPositiveCounts = new ConcurrentDictionary<string, int>(combinedPositiveCounts, StringComparer.Ordinal),
            ImageTagRowsByCacheRow = existingImages.ToDictionary(
                e => e.CacheRowIndex,
                e => new HashSet<int>(committedTagRows[e.CacheRowIndex])),
            // Always starts fresh, even on a resumed run — crawl.sqlite only persists
            // the combined-across-sites count, not a per-site breakdown, so there's
            // nothing durable to seed this from. Worst case on resume: the one tag each
            // site was mid-page on when the process last stopped searches a bit more
            // (or less) against its floor than it "truly" needed to — a minor
            // imprecision, not a correctness issue, since a genuinely-exhausted tag's
            // per-(tag,site,phase) Done flag still short-circuits the loop regardless.
            SitePositiveCounts = sites.ToDictionary(site => site, _ => new ConcurrentDictionary<string, int>(StringComparer.Ordinal), StringComparer.Ordinal),
        };

        // Guards every touch of state/vocabulary/writer once a post's dedup check needs
        // to commit something — see this class's own doc comment for why the download
        // itself deliberately happens outside this lock.
        using var stateLock = new SemaphoreSlim(1, 1);
        var pageCounter = 0;

        // Deliberately CancellationToken.None throughout this method, never the run's
        // own token: once a checkpoint starts, Flush() has already made the cache file
        // durable, and the matching CommitPendingImagesAsync MUST follow through no
        // matter what — a cancellation landing in between (e.g. Ctrl+C mid-checkpoint)
        // used to let CommitPendingImagesAsync throw OperationCanceledException before
        // it even started, leaving the cache ahead of crawl.sqlite with no record of
        // what it just durably wrote. Observed for real: a live dataset with 557 images
        // sitting in images.bin/tag_rows.jsonl that crawl.sqlite had never heard of,
        // traced to exactly this window. The *next* page fetch/download still honors
        // the real token normally (see RunSiteTagPhaseAsync) — this only shields the
        // narrow, already-in-flight commit, not the whole run.
        async Task CheckpointAsync(SiteProgressReporter? siteReporter)
        {
            await stateLock.WaitAsync(CancellationToken.None).ConfigureAwait(false);
            try
            {
                siteReporter?.ReportPhase("Checkpointing (cache, vocabulary, dedup index)...");

                // Order matters for crash-consistency: the cache/vocabulary files must be
                // durable before the dedup index is, and the dedup index durable before the
                // pagination cursor advances past the posts it covers — see this class's own
                // doc comment and CommitPendingImagesAsync's for exactly what a crash between
                // any two of these costs (always just re-fetched/duplicated work, never a
                // silently-corrupted resume). MergeTagRows must run after Flush (it only
                // touches rows Flush just made durable) and before CommitPendingImagesAsync
                // (whose CombinedPositiveCount bump for a merge should only become durable
                // once the label file actually reflects it).
                vocabulary.SaveDelta(vocabularyDeltaPath);
                writer.Flush();

                if (state.DirtyCacheRows.Count > 0)
                {
                    writer.MergeTagRows(state.DirtyCacheRows.ToDictionary(
                        row => row,
                        IReadOnlyList<int> (row) => state.ImageTagRowsByCacheRow[row].ToList()));
                    state.DirtyCacheRows.Clear();
                }

                if (state.PendingNewImages.Count > 0 || state.PendingAdditionalSources.Count > 0 || state.PendingMergedTagCounts.Count > 0)
                {
                    await db.CommitPendingImagesAsync(state.PendingNewImages, state.PendingAdditionalSources, state.PendingMergedTagCounts, CancellationToken.None).ConfigureAwait(false);
                    state.PendingNewImages.Clear();
                    state.PendingAdditionalSources.Clear();
                    state.PendingMergedTagCounts.Clear();
                }

                pageCounter++;
                if (pageCounter % vocabCompactIntervalPages == 0)
                {
                    vocabulary.Save(vocabularyPath);
                    File.Delete(vocabularyDeltaPath);
                }
            }
            finally
            {
                stateLock.Release();
            }
        }

        var shortfalls = new List<TagShortfall>();

        // Guarantees each site a fair shot at contributing something of its own before
        // a shared quota can close a tag out from under it. Without this, a site that's
        // simply faster (more requests/sec, a bigger page size) systematically wins the
        // race to satisfy CombinedPositiveCounts on every tag with meaningful overlap,
        // and the slower site's own pagination never gets far enough to reach content
        // the faster site doesn't already have — observed for real on a live crawl as
        // zero images across a 15,944-image corpus where gelbooru was the sole source,
        // despite gelbooru's own requests clearly going out and its own tags-completed
        // count climbing the whole time. Each site keeps searching a tag until EITHER it
        // personally accounts for its own even share of --max-images, OR the site
        // genuinely runs out of its own posts for that tag (Done) — regardless of how
        // far ahead of --max-images the *combined* count from other sites already is.
        // Trade-off: a tag can end up with more than --max-images total once every site
        // insists on searching for its own floor, since floors aren't reduced by what
        // other sites already found.
        var perSiteFloor = (maxImages + sites.Count - 1) / sites.Count;

        // One self-contained worker per site: its own full positive pass over every
        // eligible tag, then its own full negative pass — no cross-site barrier between
        // the two phases (see this class's own doc comment for why: a site that never
        // gives up retrying plus a hard barrier would deadlock the whole run the moment
        // one site had a real outage). The tradeoff is that a fast site's negative
        // top-up can start before a slow (or currently-retrying) site finishes
        // contributing to the shared corpus, so its negative-shortfall arithmetic sees
        // a smaller-than-final total image count and may pull a few more negatives than
        // strictly needed — a minor quality cost against the alternative of the run
        // being able to hang indefinitely.
        async Task RunSiteWorkerAsync(string site)
        {
            var siteReporter = progress?.Sites.GetValueOrDefault(site);
            var client = clientsBySite[site];
            var mySitePositiveCounts = state.SitePositiveCounts[site];

            siteReporter?.ReportTagsCompleted(0, eligibleTags.Count);
            var tagIndex = 0;
            foreach (var tag in eligibleTags)
            {
                tagIndex++;
                cancellationToken.ThrowIfCancellationRequested();

                siteReporter?.ReportPhase($"{PositivePhase} crawl: tag '{tag.Name}' ({tagIndex}/{eligibleTags.Count} eligible)");

                await RunSiteTagPhaseAsync(
                    site, client, PositivePhase, tag.Name, tag.Name,
                    () => mySitePositiveCounts.GetValueOrDefault(tag.Name) < perSiteFloor
                          || CrawlQuota.ShouldContinueFetching(state.CombinedPositiveCount(tag.Name), maxImages),
                    db, vocabulary, writer, eligibleTagSet, tempDir, inputSize, downloadClient,
                    state, stateLock, siteReporter, progress?.ReportOverall, estimatedTotal, CheckpointAsync,
                    errorLog, retryDelay, () => mySitePositiveCounts.GetValueOrDefault(tag.Name), perSiteFloor, cancellationToken).ConfigureAwait(false);

                siteReporter?.ReportTagsCompleted(tagIndex, eligibleTags.Count);
            }

            siteReporter?.ReportTagsCompleted(0, eligibleTags.Count);
            tagIndex = 0;
            foreach (var tag in eligibleTags)
            {
                tagIndex++;
                cancellationToken.ThrowIfCancellationRequested();

                var negativeQuery = $"-{tag.Name}";
                siteReporter?.ReportPhase($"{NegativePhase} crawl: tag '{tag.Name}' ({tagIndex}/{eligibleTags.Count})");

                await RunSiteTagPhaseAsync(
                    site, client, NegativePhase, tag.Name, negativeQuery,
                    () => CrawlQuota.NegativeShortfall(writer.ImageCount, state.CombinedPositiveCount(tag.Name), negativeTarget) > 0,
                    db, vocabulary, writer, eligibleTagSet, tempDir, inputSize, downloadClient,
                    state, stateLock, siteReporter, progress?.ReportOverall, estimatedTotal, CheckpointAsync,
                    errorLog, retryDelay, () => writer.ImageCount - state.CombinedPositiveCount(tag.Name), negativeTarget, cancellationToken).ConfigureAwait(false);

                siteReporter?.ReportTagsCompleted(tagIndex, eligibleTags.Count);
            }
        }

        try
        {
            progress?.ReportOverall(writer.ImageCount, Math.Max(estimatedTotal, writer.ImageCount));

            await Task.WhenAll(sites.Select(RunSiteWorkerAsync)).ConfigureAwait(false);

            foreach (var tag in eligibleTags)
            {
                var finalCount = state.CombinedPositiveCount(tag.Name);
                if (finalCount < maxImages)
                    shortfalls.Add(new TagShortfall(tag.Name, finalCount, maxImages));
            }

            vocabulary.Save(vocabularyPath);
            if (File.Exists(vocabularyDeltaPath))
                File.Delete(vocabularyDeltaPath);
            writer.Flush();

            return new CrawlResult(shortfalls);
        }
        finally
        {
            progress?.Dispose();
        }
    }

    /// <summary>Backoff between a site failure and retrying the exact same page — see this class's own doc comment for why a site never gives up outright.</summary>
    internal static readonly TimeSpan SiteRetryDelay = TimeSpan.FromMinutes(20);

    /// <summary>
    /// Logs <paramref name="ex"/> to <paramref name="errorLog"/>, puts a clear
    /// "ERROR ... retrying at HH:mm:ss" status on <paramref name="siteReporter"/>'s own
    /// phase row (left in place for the whole wait, since nothing else touches that
    /// site's row while it's asleep), and waits <see cref="SiteRetryDelay"/> before
    /// returning so the caller can retry the same operation.
    /// </summary>
    private static async Task HandleSiteFailureAsync(
        string site,
        Exception ex,
        CrawlErrorLog errorLog,
        SiteProgressReporter? siteReporter,
        Func<TimeSpan, CancellationToken, Task> retryDelay,
        CancellationToken cancellationToken)
    {
        errorLog.Log(site, ex.Message);

        var retryAt = DateTimeOffset.Now + SiteRetryDelay;
        siteReporter?.ReportPhase($"ERROR: {ex.Message} — retrying at {retryAt:HH:mm:ss} ({SiteRetryDelay.TotalMinutes:0} min)");

        await retryDelay(SiteRetryDelay, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// One site's dedicated worker for one tag/phase: fetches pages for
    /// <paramref name="tagQuery"/> from <paramref name="site"/> only (no more picking
    /// between sites — each site is its own concurrent worker now, see this class's own
    /// doc comment), processing each post and checkpointing every page, until either
    /// <paramref name="shouldContinue"/> says the target is met or this site's own
    /// pagination for this tag/phase is exhausted.
    ///
    /// A page fetch or post download that fails outright (not the transient blips
    /// <see cref="TransientHttpRetry"/> already retries — this is what's left once that's
    /// exhausted) never gives up on this site: <see cref="HandleSiteFailureAsync"/> logs
    /// it, shows it, waits, and this method retries the exact same page — the pagination
    /// cursor is never advanced past a page that didn't fully succeed, so a retry (or a
    /// resumed run after a crash) can't skip anything.
    /// </summary>
    private static async Task RunSiteTagPhaseAsync(
        string site,
        IBooruClient client,
        string phase,
        string progressTagName,
        string tagQuery,
        Func<bool> shouldContinue,
        CrawlDatabase db,
        TagVocabulary vocabulary,
        PreprocessedDatasetCacheWriter writer,
        HashSet<string> eligibleTagSet,
        string tempDir,
        int inputSize,
        HttpClient downloadClient,
        CrawlWorkingState state,
        SemaphoreSlim stateLock,
        SiteProgressReporter? siteReporter,
        Action<long, long>? reportOverall,
        long estimatedTotal,
        Func<SiteProgressReporter?, Task> checkpointAsync,
        CrawlErrorLog errorLog,
        Func<TimeSpan, CancellationToken, Task> retryDelay,
        Func<int> currentTagProgress,
        int tagProgressTarget,
        CancellationToken cancellationToken)
    {
        siteReporter?.ReportTagProgress(progressTagName, currentTagProgress(), tagProgressTarget);

        // Only the positive phase searches FOR a specific tag — the negative phase's
        // tagQuery excludes it, so a post landing here says nothing about whether this
        // site should get fairness-floor credit for progressTagName (see ProcessPostAsync).
        var searchedTag = phase == PositivePhase ? progressTagName : null;

        var tagProgress = await db.GetTagProgressAsync(progressTagName, site, phase, cancellationToken).ConfigureAwait(false);

        while (!tagProgress.Done && shouldContinue())
        {
            cancellationToken.ThrowIfCancellationRequested();

            BooruPostPage page;
            try
            {
                siteReporter?.ReportPhase($"{phase} crawl: tag '{progressTagName}', fetching page...");
                page = await client.ListPostsAsync(tagQuery, tagProgress.Cursor, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                await HandleSiteFailureAsync(site, ex, errorLog, siteReporter, retryDelay, cancellationToken).ConfigureAwait(false);
                continue; // retry the same page fetch now that the cooldown's passed
            }

            var pageFailed = false;
            siteReporter?.ReportPhase($"{phase} crawl: tag '{progressTagName}', processing {page.Posts.Count} posts...");
            for (var i = 0; i < page.Posts.Count; i++)
            {
                if (!shouldContinue())
                    break;

                siteReporter?.ReportPhaseProgress(i, page.Posts.Count);

                bool appended;
                try
                {
                    appended = await ProcessPostAsync(
                        page.Posts[i], site, searchedTag, client.RateLimiter, vocabulary, writer, eligibleTagSet, tempDir, inputSize, downloadClient, state, stateLock, cancellationToken).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    // Exit requested mid-page: whatever posts already got appended this
                    // page must be durably committed before we honor it, or the cache
                    // file (made durable by this same checkpoint's own Flush call) ends
                    // up ahead of crawl.sqlite with no record of what it just wrote —
                    // exactly the gap a live dataset hit for real. checkpointAsync
                    // itself can't be cut short by this same cancellation (see its own
                    // doc comment), so this reliably finishes before the exit proceeds.
                    await checkpointAsync(siteReporter).ConfigureAwait(false);
                    throw;
                }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    // The rest of this page is presumed unreachable too (same dead
                    // site/CDN) — stop processing it rather than retrying each
                    // remaining post individually against a site that's already down.
                    await HandleSiteFailureAsync(site, ex, errorLog, siteReporter, retryDelay, cancellationToken).ConfigureAwait(false);
                    pageFailed = true;
                    break;
                }

                if (appended)
                    reportOverall?.Invoke(writer.ImageCount, Math.Max(estimatedTotal, writer.ImageCount));

                // Unconditional, not just on a fresh append: a dedup-matched duplicate
                // can still credit this tag via ProcessPostAsync's merge path (a known
                // image gaining a tag it didn't have before) without appended being
                // true — gating this on appended alone left the bar showing a stale
                // starting count through however many merge-only credits happened
                // since, which is exactly what made it look frozen at 0 on a page full
                // of already-known images.
                siteReporter?.ReportTagProgress(progressTagName, currentTagProgress(), tagProgressTarget);
            }
            siteReporter?.ReportPhaseProgress(page.Posts.Count, page.Posts.Count);

            // Checkpoint (durably commit this page's work) before advancing the cursor
            // past it — see this file's class-level doc comment on why that ordering is
            // what makes a crash mid-run recoverable instead of silently lossy. Applies
            // even when the page died partway through: whatever posts did get appended
            // before the failure are durable either way.
            await checkpointAsync(siteReporter).ConfigureAwait(false);

            if (pageFailed)
                continue; // retry the same page (cursor wasn't advanced) now that the cooldown's passed

            // CancellationToken.None: the checkpoint just above already committed this
            // page's images durably — advancing the cursor to match is the other half
            // of that same atomic unit of work, not a new one an exit request should be
            // able to cut off. Worst case otherwise is mild (this page gets refetched
            // and its now-known posts correctly recognized as duplicates, not orphaned),
            // but there's no reason not to close this out cleanly too.
            var done = page.NextCursor is null;
            var nextTagProgress = new TagProgressState(page.NextCursor, tagProgress.PostsFetched + page.Posts.Count, done);
            await db.SaveTagProgressAsync(progressTagName, site, phase, nextTagProgress, CancellationToken.None).ConfigureAwait(false);
            tagProgress = nextTagProgress;
        }
    }

    /// <summary>
    /// Dedup-skips a known image (exact md5 or perceptual near-duplicate — buffers only
    /// an additional-source record), or downloads+normalizes+appends a new one (buffers
    /// a full pending image record). Nothing here talks to <c>crawl.sqlite</c> directly —
    /// everything durable is deferred to the caller's next checkpoint. Returns whether a
    /// new image was actually appended (for overall-progress bookkeeping) — any dedup
    /// skip or an undecodable download return false.
    ///
    /// Only the fast dedup-check-and-commit parts run under <paramref name="stateLock"/>;
    /// the slow part (the network download and phash/normalize work) deliberately runs
    /// unlocked so two sites' downloads can actually overlap. That means a post can be
    /// checked "not yet known", downloaded, and only THEN find another site's worker
    /// already committed the exact same image (or a near-duplicate) while this one's
    /// download was in flight — the lock is re-acquired and every check re-run right
    /// before the final commit specifically to catch that race, rather than trusting the
    /// stale answer from the first check.
    /// </summary>
    private static async Task<bool> ProcessPostAsync(
        BooruPost post,
        string site,
        string? searchedTag,
        IRateLimiter rateLimiter,
        TagVocabulary vocabulary,
        PreprocessedDatasetCacheWriter writer,
        HashSet<string> eligibleTagSet,
        string tempDir,
        int inputSize,
        HttpClient downloadClient,
        CrawlWorkingState state,
        SemaphoreSlim stateLock,
        CancellationToken cancellationToken)
    {
        await stateLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (state.KnownImages.TryGetValue(post.Md5, out _))
            {
                var observedTags = MergeDuplicateTags(post, post.Md5, site, searchedTag, vocabulary, eligibleTagSet, state);
                state.PendingAdditionalSources.Add(new PendingAdditionalSource(
                    post.Md5, site, post.PostId, post.FileUrl.ToString(), post.Rating, post.CreatedAt, observedTags, DateTimeOffset.UtcNow));
                return false;
            }
        }
        finally
        {
            stateLock.Release();
        }

        var tempPath = Path.Combine(tempDir, $"{Guid.NewGuid():N}.img");
        try
        {
            // Raw image bytes are typically served from the same site/CDN as the JSON
            // API, so reuse its rate limiter here too — this used to be a completely
            // unthrottled GetStreamAsync, which fired a whole page's worth of downloads
            // back-to-back with no pacing at all and no retry, so a single 429 from the
            // CDN crashed the entire run instead of just backing off.
            //
            // Referer set to the file's own origin on every request: Gelbooru's CDN
            // enforces hotlink protection (a bare request — no Referer at all — gets
            // 302'd to gelbooru.com/hotlink.php, which itself redirects to the normal
            // HTML post page; EnsureSuccessStatusCode below doesn't catch this since
            // the final response is a real 200, just of the wrong content, so it silently
            // looked like a corrupt/undecodable file with no indication a real image was
            // ever reachable). Confirmed empirically: either the CDN host itself or the
            // main site as Referer satisfies Gelbooru's check; Danbooru doesn't require
            // one at all but is unaffected by always sending it, so no per-site branch
            // is needed here.
            using var response = await TransientHttpRetry.SendWithRetryAsync(
                async () =>
                {
                    await rateLimiter.WaitAsync(cancellationToken).ConfigureAwait(false);
                    using var request = new HttpRequestMessage(HttpMethod.Get, post.FileUrl);
                    request.Headers.Referrer = new Uri($"{post.FileUrl.Scheme}://{post.FileUrl.Host}/");
                    return await downloadClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
                },
                post.FileUrl,
                cancellationToken).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            await using (var fileStream = File.Create(tempPath))
            await using (var responseStream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
            {
                await responseStream.CopyToAsync(fileStream, cancellationToken).ConfigureAwait(false);
            }

            ulong phash;
            float[] pixels;
            try
            {
                phash = PerceptualHash.Compute(tempPath);
                pixels = ImagePreprocessing.LoadAndNormalize(tempPath, inputSize);
            }
            catch (InvalidDataException)
            {
                // Corrupt/unsupported download — skip this post entirely rather than
                // recording a broken image or crashing the whole crawl run.
                return false;
            }

            await stateLock.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                // Re-check everything: another site's worker may have committed this
                // exact post (or a near-duplicate) while this download was in flight.
                if (state.KnownImages.TryGetValue(post.Md5, out _))
                {
                    var observedTags = MergeDuplicateTags(post, post.Md5, site, searchedTag, vocabulary, eligibleTagSet, state);
                    state.PendingAdditionalSources.Add(new PendingAdditionalSource(
                        post.Md5, site, post.PostId, post.FileUrl.ToString(), post.Rating, post.CreatedAt, observedTags, DateTimeOffset.UtcNow));
                    return false;
                }

                var duplicate = state.HashIndex.FirstOrDefault(entry => PerceptualHash.HammingDistance(entry.Hash, phash) <= MaxHammingDistance);
                if (duplicate.Md5 is not null)
                {
                    // Same artwork, re-encoded by this site — attribute it as another source
                    // of the already-cached (canonical) image rather than appending a
                    // near-identical duplicate under a different md5.
                    var observedTags = MergeDuplicateTags(post, duplicate.Md5, site, searchedTag, vocabulary, eligibleTagSet, state);
                    state.PendingAdditionalSources.Add(new PendingAdditionalSource(
                        duplicate.Md5, site, post.PostId, post.FileUrl.ToString(), post.Rating, post.CreatedAt, observedTags, DateTimeOffset.UtcNow));
                    return false;
                }

                var eligibleTagsOnPost = post.Tags.Where(eligibleTagSet.Contains).Distinct(StringComparer.Ordinal).ToList();
                var tagRows = new List<int>(eligibleTagsOnPost.Count);
                foreach (var tagName in eligibleTagsOnPost)
                {
                    var record = vocabulary.RecordObservation(tagName);
                    tagRows.Add(record.RowIndex);
                }

                writer.Append(pixels, tagRows);
                var cacheRowIndex = writer.ImageCount - 1;

                state.KnownImages[post.Md5] = cacheRowIndex;
                state.HashIndex.Add((phash, post.Md5));
                state.ImageTagRowsByCacheRow[cacheRowIndex] = new HashSet<int>(tagRows);
                var mySitePositiveCounts = state.SitePositiveCounts[site];
                foreach (var tagName in eligibleTagsOnPost)
                {
                    state.CombinedPositiveCounts[tagName] = state.CombinedPositiveCount(tagName) + 1;
                    mySitePositiveCounts[tagName] = mySitePositiveCounts.GetValueOrDefault(tagName) + 1;
                }

                state.PendingNewImages.Add(new PendingNewImage(
                    post.Md5, cacheRowIndex, post.Width, post.Height, DateTimeOffset.UtcNow,
                    site, post.PostId, post.FileUrl.ToString(), post.Rating, post.CreatedAt,
                    eligibleTagsOnPost, phash));

                return true;
            }
            finally
            {
                stateLock.Release();
            }
        }
        finally
        {
            if (File.Exists(tempPath))
                File.Delete(tempPath);
        }
    }

    /// <summary>
    /// A duplicate post (matched by exact md5 or perceptual hash) can still carry tags
    /// its earlier-seen copy from the other site didn't — this project's whole reason
    /// for crawling two sites per tag in the first place is that neither site's tagging
    /// is complete on its own. Folds any such tags into the canonical row's live
    /// in-memory tag set and marks the row dirty for the next checkpoint's
    /// <see cref="PreprocessedDatasetCacheWriter.MergeTagRows"/> call; a no-op when the
    /// duplicate's tags are already a subset of what the image already has (the common
    /// case — most cross-site duplicates agree). Always called while holding
    /// <c>stateLock</c> — see <see cref="ProcessPostAsync"/>.
    /// Returns this post's own eligible tags (not just the newly-added ones) so the
    /// caller can record them as this source's fresh snapshot — the ingredient
    /// <c>refresh-tags</c> later reconciles across every source of an image, including
    /// possible removal (see <see cref="TagRefresher"/>); this additive merge itself
    /// never removes anything, since it can't afford to check every other source mid-crawl.
    ///
    /// <paramref name="site"/>'s own <see cref="CrawlWorkingState.SitePositiveCounts"/>
    /// entry is credited for each newly-added tag too: this site's copy demonstrably
    /// carried a tag an earlier-seen copy of the same image didn't, which is exactly the
    /// kind of find the per-site fairness floor (see <see cref="RunAsync"/>) exists to
    /// recognize — it's not "catching up" to something already known, it's this site
    /// answering a question about the image the other one's copy didn't.
    ///
    /// Separately, <paramref name="searchedTag"/> (the tag this site's positive-phase
    /// search is actually FOR, or null during the negative phase — see
    /// <see cref="RunSiteTagPhaseAsync"/>) also gets one fairness-floor credit here even
    /// when it's already on the canonical image and so ISN'T in the newly-added set
    /// above: this site still did real, correct work returning a genuinely matching
    /// post, and CombinedPositiveCounts must NOT also move for it (that field means
    /// "unique tag-image associations", and this isn't a new one). Without this, a site
    /// whose search results page happens to be entirely redundant with what's already
    /// known gets zero credit for any of it and grinds through its own remaining
    /// pagination stuck at 0/floor — indistinguishable from that site not trying at all.
    /// </summary>
    private static IReadOnlyList<string> MergeDuplicateTags(
        BooruPost post,
        string canonicalMd5,
        string site,
        string? searchedTag,
        TagVocabulary vocabulary,
        HashSet<string> eligibleTagSet,
        CrawlWorkingState state)
    {
        var cacheRowIndex = state.KnownImages[canonicalMd5];
        var currentTagRows = state.ImageTagRowsByCacheRow[cacheRowIndex];

        var observedTags = post.Tags.Where(eligibleTagSet.Contains).Distinct(StringComparer.Ordinal).ToList();

        List<string>? newlyAddedTags = null;
        foreach (var tagName in observedTags)
        {
            if (TagRowMutations.TryAddTagToImage(vocabulary, tagName, currentTagRows))
                (newlyAddedTags ??= []).Add(tagName);
        }

        var mySitePositiveCounts = state.SitePositiveCounts[site];

        if (newlyAddedTags is not null)
        {
            state.DirtyCacheRows.Add(cacheRowIndex);
            foreach (var tagName in newlyAddedTags)
            {
                state.PendingMergedTagCounts.Add(tagName);
                state.CombinedPositiveCounts[tagName] = state.CombinedPositiveCount(tagName) + 1;
                mySitePositiveCounts[tagName] = mySitePositiveCounts.GetValueOrDefault(tagName) + 1;
            }
        }

        if (searchedTag is not null
            && (newlyAddedTags is null || !newlyAddedTags.Contains(searchedTag, StringComparer.Ordinal))
            && observedTags.Contains(searchedTag, StringComparer.Ordinal))
        {
            mySitePositiveCounts[searchedTag] = mySitePositiveCounts.GetValueOrDefault(searchedTag) + 1;
        }

        return observedTags;
    }
}
