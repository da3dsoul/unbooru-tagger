using System.Collections.Concurrent;
using System.Net;
using SkiaSharp;
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
    public required PerceptualHashIndex HashIndex { get; init; }
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
    /// needed to add/remove a site's own entry later. Each site's dictionary starts empty
    /// (not persisted directly), but <see cref="DatasetCrawler.RunSiteTagPhaseAsync"/>
    /// seeds the one entry that actually needs it — whatever tag that site is resuming
    /// mid-pagination into — from durable history before relying on it.
    /// </summary>
    public required IReadOnlyDictionary<string, ConcurrentDictionary<string, int>> SitePositiveCounts { get; init; }

    /// <summary>
    /// Per (site, tag), how many of that site's own duplicate finds — a post that turned
    /// out to already be known, by exact md5 or perceptual-hash near-match — matched the
    /// tag currently being searched. Purely a diagnostic surfaced on the "Current tag"
    /// progress row (see <see cref="DatasetCrawler.RunSiteTagPhaseAsync"/>) so it's
    /// visible whether a tag's overshoot past <c>--max-images</c> is duplicate-driven;
    /// never read by any quota decision. Same empty-per-site-dictionary start and
    /// resume-seed-only-the-current-tag pattern as <see cref="SitePositiveCounts"/> — a
    /// resumed run seeds straight from <see cref="TagProgressState.SiteDuplicateCount"/>,
    /// a durable per-page checkpoint of this same dictionary's value.
    /// </summary>
    public required IReadOnlyDictionary<string, ConcurrentDictionary<string, int>> SiteDuplicateCounts { get; init; }

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
/// scratch file just long enough to decode/resize via
/// <see cref="ImagePreprocessing.LoadAndEncode(string, int)"/> and append to the same
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
/// part (the network download and CPU-bound decode/hash/resize) deliberately happens
/// outside the lock, split into its own download and processing phases (see
/// <see cref="DownloadPostAsync"/>/<see cref="ProcessDownloadedPostAsync"/>) each gated
/// by their own per-site worker pool — see the check-download-recheck pattern documented
/// there for what keeps that safe against two sites (or two posts on the same site)
/// discovering the same image at nearly the same moment.
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
        int tagSurveyRequestsMade,
        TagExclusionRules? excludedTags = null)
    {
        var eligibleTags = allTags.Where(t => TagEligibility.IsEligible(t, minImages, excludedTags)).ToList();
        var imageSlots = TagEligibility.EstimateImageSlots(eligibleTags, minImages, maxImages, excludedTags);

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
        TagExclusionRules? excludedTags = null,
        Func<TimeSpan, CancellationToken, Task>? retryDelay = null,
        IReadOnlyDictionary<string, string>? tagAliases = null)
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
        var eligibleTags = CrawlScheduling.RarestFirst(allTags.Where(t => TagEligibility.IsEligible(t, minImages, excludedTags))).ToList();
        var estimatedTotal = TagEligibility.EstimateImageSlots(eligibleTags, minImages, maxImages, excludedTags);

        ReportSetupPhase("Loading tag vocabulary...");
        var vocabularyPath = Path.Combine(outputDirectory, "tag_vocabulary.json");
        var vocabularyDeltaPath = Path.Combine(outputDirectory, "tag_vocabulary.delta.jsonl");
        var vocabulary = File.Exists(vocabularyPath)
            ? TagVocabulary.Load(vocabularyPath, vocabularyDeltaPath)
            : TagVocabulary.CreateEmpty();
        if (!File.Exists(vocabularyPath))
            vocabulary.Save(vocabularyPath);

        // Keyed by raw (un-prefixed) booru name, not identity — the only form a post's
        // tags ever come back as. See TagRowMutations.BuildEligibleIdentities — tagAliases
        // here matters because Gelbooru will keep returning an aliased-away raw spelling
        // (e.g. head_pat) on its own posts forever regardless of what Danbooru calls it,
        // so without this the tag would silently stop being recorded at all rather than
        // correctly crediting the merged identity.
        var eligibleTagIdentities = TagRowMutations.BuildEligibleIdentities(eligibleTags, tagAliases);

        var tempDir = Path.Combine(outputDirectory, ".tmp");
        Directory.CreateDirectory(tempDir);
        foreach (var leftover in Directory.EnumerateFiles(tempDir))
            File.Delete(leftover); // safe: nothing durable is recorded until after a successful checkpoint

        ReportSetupPhase("Opening cache writer...");
        using var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(outputDirectory, inputSize,
            onResumeProgress: (subPhase, completed, total) => ReportSetupPhase($"Opening cache writer: {subPhase} ({completed}/{total})..."));

        // Seeded once from durable state (accurate as of the last successful checkpoint,
        // exactly in step with the cache file writer above thanks to its own
        // truncate-to-last-flush resume logic), then kept live in memory from here on —
        // see CrawlWorkingState's own doc comment for why the durable copies deliberately
        // lag behind these during a run.
        ReportSetupPhase("Loading dedup index and tag counters...");
        var existingImages = await db.GetAllImagesAsync(cancellationToken,
            onProgress: completed => ReportSetupPhase($"Loading dedup index and tag counters: images loaded ({completed})...")).ConfigureAwait(false);
        CacheConsistency.Validate(existingImages, writer.ImageCount, outputDirectory);
        var combinedPositiveCounts = await db.GetAllCombinedPositiveCountsAsync(cancellationToken).ConfigureAwait(false);
        var committedTagRows = writer.ReadCommittedTagRows(
            onProgress: (completed, total) => ReportSetupPhase($"Loading dedup index and tag counters: tag rows ({completed}/{total})..."));
        var state = new CrawlWorkingState
        {
            KnownImages = existingImages.ToDictionary(e => e.Md5, e => e.CacheRowIndex, StringComparer.Ordinal),
            HashIndex = new PerceptualHashIndex(MaxHammingDistance, existingImages.Select(e => (e.PHash, e.Md5))),
            CombinedPositiveCounts = new ConcurrentDictionary<string, int>(combinedPositiveCounts, StringComparer.Ordinal),
            ImageTagRowsByCacheRow = existingImages.ToDictionary(
                e => e.CacheRowIndex,
                e => new HashSet<int>(committedTagRows[e.CacheRowIndex])),
            // Starts empty here, not fresh-and-staying-that-way: RunSiteTagPhaseAsync
            // seeds just the ONE entry that actually needs it — the tag a site's worker
            // is resuming mid-pagination into — directly from that (tag, site, phase)
            // row's own durable SitePositiveCount/SiteDuplicateCount (crawl.sqlite's
            // TagProgress table) the moment it's about to become "current tag", rather
            // than bulk-loading every tag's row up front. Every other tag's real answer
            // already is 0 (never touched, or a genuinely-exhausted/quota-satisfied tag
            // whose Done/QuotaSatisfiedAtMaxImages short-circuits the loop before this
            // dictionary is even consulted), so leaving them unseeded costs nothing.
            SitePositiveCounts = sites.ToDictionary(site => site, _ => new ConcurrentDictionary<string, int>(StringComparer.Ordinal), StringComparer.Ordinal),
            SiteDuplicateCounts = sites.ToDictionary(site => site, _ => new ConcurrentDictionary<string, int>(StringComparer.Ordinal), StringComparer.Ordinal),
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

            // One pair of pools per site, not shared globally: a site's own downloads
            // and decode/hash/resize work should never queue up behind the OTHER site's
            // work. With one shared pool, danbooru (the faster site, higher rate limit)
            // could fill it and leave gelbooru's already-slow downloads waiting even
            // longer for CPU time — compounding its rate-limit disadvantage instead of
            // just living with it. Both sized off logical processor count; some
            // oversubscription across sites running at once is possible (2 sites x N
            // cores of processing headroom each), but gelbooru's own rate limit is what
            // actually keeps its real concurrent usage low in practice.
            using var downloadPool = new SemaphoreSlim(Environment.ProcessorCount, Environment.ProcessorCount);
            using var processingPool = new SemaphoreSlim(Environment.ProcessorCount, Environment.ProcessorCount);

            siteReporter?.ReportTagsCompleted(0, eligibleTags.Count);
            var tagIndex = 0;
            foreach (var tag in eligibleTags)
            {
                tagIndex++;
                cancellationToken.ThrowIfCancellationRequested();

                siteReporter?.ReportPhase($"{PositivePhase} crawl: tag '{tag.Name}' ({tagIndex}/{eligibleTags.Count} eligible)");

                // The site's own search API only ever understands a tag's raw booru
                // name, never its category-prefixed identity — every other argument
                // here (progress display, quota bookkeeping) uses the identity.
                await RunSiteTagPhaseAsync(
                    site, client, PositivePhase, tag.Name, TagCategoryNaming.RawName(tag.Name),
                    () => mySitePositiveCounts.GetValueOrDefault(tag.Name) < perSiteFloor
                          || CrawlQuota.ShouldContinueFetching(state.CombinedPositiveCount(tag.Name), maxImages),
                    db, vocabulary, writer, eligibleTagIdentities, tempDir, inputSize, downloadClient,
                    state, stateLock, downloadPool, processingPool, siteReporter, progress?.ReportOverall, estimatedTotal, CheckpointAsync,
                    errorLog, retryDelay, () => mySitePositiveCounts.GetValueOrDefault(tag.Name), perSiteFloor,
                    () => state.SiteDuplicateCounts[site].GetValueOrDefault(tag.Name), maxImages, cancellationToken).ConfigureAwait(false);

                siteReporter?.ReportTagsCompleted(tagIndex, eligibleTags.Count);
            }

            siteReporter?.ReportTagsCompleted(0, eligibleTags.Count);
            tagIndex = 0;
            foreach (var tag in eligibleTags)
            {
                tagIndex++;
                cancellationToken.ThrowIfCancellationRequested();

                var negativeQuery = $"-{TagCategoryNaming.RawName(tag.Name)}";
                siteReporter?.ReportPhase($"{NegativePhase} crawl: tag '{tag.Name}' ({tagIndex}/{eligibleTags.Count})");

                await RunSiteTagPhaseAsync(
                    site, client, NegativePhase, tag.Name, negativeQuery,
                    () => CrawlQuota.NegativeShortfall(writer.ImageCount, state.CombinedPositiveCount(tag.Name), negativeTarget) > 0,
                    db, vocabulary, writer, eligibleTagIdentities, tempDir, inputSize, downloadClient,
                    state, stateLock, downloadPool, processingPool, siteReporter, progress?.ReportOverall, estimatedTotal, CheckpointAsync,
                    errorLog, retryDelay, () => writer.ImageCount - state.CombinedPositiveCount(tag.Name), negativeTarget,
                    () => 0, maxImagesForQuotaTracking: null, cancellationToken).ConfigureAwait(false);

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
        IReadOnlyDictionary<string, string> eligibleTagIdentities,
        string tempDir,
        int inputSize,
        HttpClient downloadClient,
        CrawlWorkingState state,
        SemaphoreSlim stateLock,
        SemaphoreSlim downloadPool,
        SemaphoreSlim processingPool,
        SiteProgressReporter? siteReporter,
        Action<long, long>? reportOverall,
        long estimatedTotal,
        Func<SiteProgressReporter?, Task> checkpointAsync,
        CrawlErrorLog errorLog,
        Func<TimeSpan, CancellationToken, Task> retryDelay,
        Func<int> currentTagProgress,
        int tagProgressTarget,
        Func<int> currentTagDuplicates,
        int? maxImagesForQuotaTracking,
        CancellationToken cancellationToken)
    {
        // Only the positive phase searches FOR a specific tag — the negative phase's
        // tagQuery excludes it, so a post landing here says nothing about whether this
        // site should get fairness-floor credit for progressTagName (see MergeDuplicateTags).
        var searchedTag = phase == PositivePhase ? progressTagName : null;

        var tagProgress = await db.GetTagProgressAsync(progressTagName, site, phase, cancellationToken).ConfigureAwait(false);

        // A tag whose quota was already confirmed satisfied under an equal-or-higher
        // --max-images in a past run is still guaranteed satisfied now (combined counts
        // only ever grow) — skip re-entering this tag at all. A run with a HIGHER
        // --max-images than what's recorded can't trust this (a smaller quota being met
        // doesn't mean a larger one is), so it falls through to the normal path below
        // for exactly those tags.
        if (phase == PositivePhase && maxImagesForQuotaTracking is int currentMaxImages
            && tagProgress.QuotaSatisfiedAtMaxImages is int satisfiedAt && satisfiedAt >= currentMaxImages)
        {
            return;
        }

        // SitePositiveCounts (the source of currentTagProgress() for the positive phase)
        // always starts this process at 0 for every tag — correct for one never touched
        // before, wrong for the one tag this site was actually mid-page on when it last
        // stopped: that credit has real, durable history, just not reflected here yet.
        // Seeded once, only for that tag (mid-pagination, not Done), directly from
        // TagProgressState's own SitePositiveCount/SiteDuplicateCount — durable
        // checkpoints written every page (see the SaveTagProgressAsync call below), not
        // re-derived from an ImageSources scan the way this used to work. Must happen
        // before the initial ReportTagProgress below, or that first paint would still
        // show the stale, pre-seed 0.
        if (phase == PositivePhase && tagProgress is { PostsFetched: > 0, Done: false })
        {
            var mySitePositiveCounts = state.SitePositiveCounts[site];
            if (!mySitePositiveCounts.ContainsKey(progressTagName))
                mySitePositiveCounts[progressTagName] = tagProgress.SitePositiveCount;

            var mySiteDuplicateCounts = state.SiteDuplicateCounts[site];
            if (!mySiteDuplicateCounts.ContainsKey(progressTagName))
                mySiteDuplicateCounts[progressTagName] = tagProgress.SiteDuplicateCount;
        }

        siteReporter?.ReportTagProgress(progressTagName, currentTagProgress(), tagProgressTarget, currentTagDuplicates());

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

            // Posts within a page now run concurrently instead of one at a time: a post's
            // download used to block the NEXT post's download until its own decode/
            // hash/resize/append finished too, which a profiler showed as most of a
            // page's wall time. Each site's own IRateLimiter (inside DownloadPostAsync)
            // still paces actual HTTP dispatch, and downloadPool/processingPool still cap
            // how many downloads and decode/hash/resize passes run at once, so this
            // doesn't exceed the rate limit or thrash the CPU — it just lets the next
            // download start while the current post's CPU-bound work is still in flight,
            // instead of gating on it. Download and processing are separate phases (see
            // DownloadPostAsync/ProcessDownloadedPostAsync) with their own progress rows,
            // so it's visible whether a slow page is stuck waiting on the network or on
            // decode/resize — the two used to be one indistinguishable "processing" bar.
            //
            // shouldContinue() is checked once per post right before it starts, using
            // whatever quota state is visible at that moment — with posts finishing out
            // of order, that check can be a little behind reality (a handful of extra
            // posts can start just before an in-flight one's completion would have
            // satisfied quota), which can now overshoot a page's remainder more than the
            // old strictly-sequential loop did. Accepted deliberately: the same kind of
            // small overshoot the per-site fairness floor already tolerates elsewhere in
            // this file, and cheap compared to serializing the whole page again just to
            // avoid it.
            var pageFailed = false;
            var pageCancelled = false;
            Exception? pageException = null;
            var downloadedCount = 0;
            var processedCount = 0;

            // Guards the read-then-display sequence for this page's three progress rows.
            // Without it, two posts finishing close together could interleave as: post A
            // reads/reports 11, post B reads/reports 12, but post A's WRITE to the
            // on-screen widget lands after post B's — the bar (and "current tag" count,
            // which re-reads live state each call) visibly regresses to a stale, lower
            // number and can appear stuck there, since nothing forces the higher value to
            // be written last. downloadedCount/processedCount/currentTagProgress() are
            // each individually correct (Interlocked/stateLock-protected), and Task.WhenAll
            // below still genuinely waits for every post — this lock only makes sure the
            // on-screen numbers reflect that correctly instead of showing whichever
            // update happened to land last by scheduling luck.
            var progressLock = new object();

            async Task ProcessOnePostAsync(BooruPost post)
            {
                try
                {
                    var tempPath = await DownloadPostAsync(
                        post, site, searchedTag, client.RateLimiter, vocabulary, eligibleTagIdentities, tempDir, downloadClient, state, stateLock, downloadPool, cancellationToken).ConfigureAwait(false);
                    lock (progressLock)
                        siteReporter?.ReportDownloadProgress(++downloadedCount, page.Posts.Count);

                    if (tempPath is not null)
                    {
                        // reportOverall fires from inside ProcessDownloadedPostAsync's own
                        // stateLock section, not here — see its doc comment for why.
                        await ProcessDownloadedPostAsync(
                            tempPath, post, site, searchedTag, vocabulary, writer, eligibleTagIdentities, inputSize, state, stateLock, processingPool, reportOverall, estimatedTotal, cancellationToken).ConfigureAwait(false);
                    }

                    lock (progressLock)
                    {
                        siteReporter?.ReportProcessingProgress(++processedCount, page.Posts.Count);

                        // Unconditional, not just on a fresh append: a dedup-matched
                        // duplicate can still credit this tag via the merge path (a known
                        // image gaining a tag it didn't have before) without an append —
                        // gating this on an append alone left the bar showing a stale
                        // starting count through however many merge-only credits happened
                        // since, which is exactly what made it look frozen at 0 on a page
                        // full of already-known images.
                        siteReporter?.ReportTagProgress(progressTagName, currentTagProgress(), tagProgressTarget, currentTagDuplicates());
                    }
                }
                catch (OperationCanceledException)
                {
                    pageCancelled = true;
                }
                catch (Exception ex)
                {
                    // Only the first failure is kept for HandleSiteFailureAsync below —
                    // several posts can fail near-simultaneously against the same dead
                    // site/CDN, and that should still cost one log entry and one backoff
                    // wait, not one per failed post.
                    pageFailed = true;
                    pageException ??= ex;
                }
            }

            siteReporter?.ReportPhase($"{phase} crawl: tag '{progressTagName}', processing {page.Posts.Count} posts...");
            siteReporter?.ReportDownloadProgress(0, page.Posts.Count);
            siteReporter?.ReportProcessingProgress(0, page.Posts.Count);

            var postTasks = new List<Task>(page.Posts.Count);
            foreach (var post in page.Posts)
            {
                if (!shouldContinue())
                    break;

                postTasks.Add(ProcessOnePostAsync(post));
            }

            await Task.WhenAll(postTasks).ConfigureAwait(false); // never throws: ProcessOnePostAsync catches everything itself

            if (pageCancelled)
            {
                // Exit requested mid-page: whatever posts already got appended this
                // page must be durably committed before we honor it, or the cache
                // file (made durable by this same checkpoint's own Flush call) ends
                // up ahead of crawl.sqlite with no record of what it just wrote —
                // exactly the gap a live dataset hit for real. checkpointAsync itself
                // can't be cut short by this same cancellation (see its own doc
                // comment), and every post task above has already reached a terminal
                // state (Task.WhenAll waited for that), so this is safe to run now.
                await checkpointAsync(siteReporter).ConfigureAwait(false);
                cancellationToken.ThrowIfCancellationRequested();
            }

            if (pageFailed)
            {
                // The rest of this page is presumed unreachable too (same dead
                // site/CDN) — one backoff-and-log covers every post that failed this
                // page, not one per post (see pageException above).
                await HandleSiteFailureAsync(site, pageException!, errorLog, siteReporter, retryDelay, cancellationToken).ConfigureAwait(false);
            }

            // Checkpoint (durably commit this page's work) before advancing the cursor
            // past it — see this file's class-level doc comment on why that ordering is
            // what makes a crash mid-run recoverable instead of silently lossy. Applies
            // even when the page failed partway through: whatever posts did get
            // appended before the failure are durable either way.
            await checkpointAsync(siteReporter).ConfigureAwait(false);

            if (pageFailed)
                continue; // retry the same page (cursor wasn't advanced) now that the cooldown's passed

            // CancellationToken.None: the checkpoint just above already committed this
            // page's images durably — advancing the cursor to match is the other half
            // of that same atomic unit of work, not a new one an exit request should be
            // able to cut off. Worst case otherwise is mild (this page gets refetched
            // and its now-known posts correctly recognized as duplicates, not orphaned),
            // but there's no reason not to close this out cleanly too.
            // currentTagProgress()/currentTagDuplicates() read this site's own live,
            // in-memory counts (mySitePositiveCounts/mySiteDuplicateCounts) — safe to
            // snapshot here since every post this page dispatched has already completed
            // (the Task.WhenAll above). Checkpointing them into TagProgressState every
            // page is what lets the NEXT resume seed straight from this row instead of
            // an ImageSources scan (see the resume-seed block above). Only meaningful
            // for the positive phase — the negative phase has no analogous per-site
            // count. QuotaSatisfiedAtMaxImages is explicitly cleared: we just fetched a
            // real page, so whatever was previously recorded there (from an earlier,
            // lower --max-images run) no longer reflects "not yet satisfied under the
            // CURRENT target" — the post-loop block below re-sets it if this page
            // boundary turns out to be where quota gets met.
            var done = page.NextCursor is null;
            var nextTagProgress = new TagProgressState(
                page.NextCursor,
                tagProgress.PostsFetched + page.Posts.Count,
                done,
                QuotaSatisfiedAtMaxImages: null,
                SitePositiveCount: phase == PositivePhase ? currentTagProgress() : 0,
                SiteDuplicateCount: phase == PositivePhase ? currentTagDuplicates() : 0);
            await db.SaveTagProgressAsync(progressTagName, site, phase, nextTagProgress, CancellationToken.None).ConfigureAwait(false);
            tagProgress = nextTagProgress;
        }

        // The loop above can exit for two different reasons: pagination genuinely
        // exhausted (tagProgress.Done, already durable via the SaveTagProgressAsync
        // inside the loop) or shouldContinue() became false because quota was already
        // met — that second case leaves tagProgress exactly as the last page-boundary
        // save (or the initial GetTagProgressAsync, if the loop never even ran a single
        // iteration) left it, with nothing recording WHY it stopped. Persisting that
        // here is what lets a future resume skip re-entering this tag at all instead of
        // just cheaply re-seeding it.
        if (phase == PositivePhase && maxImagesForQuotaTracking is int recordedMaxImages && !tagProgress.Done)
        {
            var quotaSatisfiedState = tagProgress with { QuotaSatisfiedAtMaxImages = recordedMaxImages };
            await db.SaveTagProgressAsync(progressTagName, site, phase, quotaSatisfiedState, CancellationToken.None).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Dedup-skips a known image (exact md5 match — buffers only an additional-source
    /// record and returns null), or downloads it to a fresh temp file and returns that
    /// path for <see cref="ProcessDownloadedPostAsync"/> to pick up. Also returns null
    /// (nothing to process, no temp file left behind) for a 404/410 — the file is
    /// permanently gone, not corrupt or worth decoding.
    ///
    /// Only the dedup pre-check runs under <paramref name="stateLock"/>; the download
    /// itself deliberately runs unlocked (gated only by <paramref name="downloadPool"/>
    /// and the site's own <paramref name="rateLimiter"/>) so downloads for many posts —
    /// across sites too — can actually overlap. A post can therefore be checked "not yet
    /// known" here and still turn out to be a duplicate (exact or near) by the time
    /// <see cref="ProcessDownloadedPostAsync"/> re-checks under the lock right before
    /// committing — that's fine: this method's whole job is producing bytes to decode,
    /// not the final word on whether they're needed.
    /// </summary>
    private static async Task<string?> DownloadPostAsync(
        BooruPost post,
        string site,
        string? searchedTag,
        IRateLimiter rateLimiter,
        TagVocabulary vocabulary,
        IReadOnlyDictionary<string, string> eligibleTagIdentities,
        string tempDir,
        HttpClient downloadClient,
        CrawlWorkingState state,
        SemaphoreSlim stateLock,
        SemaphoreSlim downloadPool,
        CancellationToken cancellationToken)
    {
        await stateLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (state.KnownImages.TryGetValue(post.Md5, out _))
            {
                var observedTags = MergeDuplicateTags(post, post.Md5, site, searchedTag, vocabulary, eligibleTagIdentities, state);
                state.PendingAdditionalSources.Add(new PendingAdditionalSource(
                    post.Md5, site, post.PostId, post.FileUrl.ToString(), post.Rating, post.CreatedAt, observedTags, DateTimeOffset.UtcNow));
                RecordDuplicateIfSearched(state, site, searchedTag);
                return null;
            }
        }
        finally
        {
            stateLock.Release();
        }

        await downloadPool.WaitAsync(cancellationToken).ConfigureAwait(false);
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

            // 404/410 mean the FILE itself is permanently gone — deleted, moved — a
            // property of this one post, not us: retrying the exact same request will
            // never succeed no matter how many times or how long it waits. Skip just
            // this post rather than letting EnsureSuccessStatusCode's exception
            // propagate up to RunSiteTagPhaseAsync's per-post catch, which treats any
            // exception as a SITE-wide failure: it aborts the rest of the page, waits
            // 20 minutes, and retries the exact same page — including this exact same
            // permanently-dead post — forever, without ever making progress past it.
            // Deliberately NOT every 4xx: 401/403 usually mean OUR request is wrong
            // (bad/expired credentials, an IP block, ...) — that's systemic and worth
            // surfacing/retrying like a real site failure, not silently swallowing
            // post after post while the actual cause goes unnoticed. A 5xx (or
            // anything else) still escalates via EnsureSuccessStatusCode below too.
            if (response.StatusCode is HttpStatusCode.NotFound or HttpStatusCode.Gone)
                return null; // no temp file created yet — nothing to clean up

            response.EnsureSuccessStatusCode();

            var tempPath = Path.Combine(tempDir, $"{Guid.NewGuid():N}.img");
            try
            {
                await using (var fileStream = File.Create(tempPath))
                await using (var responseStream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false))
                {
                    await responseStream.CopyToAsync(fileStream, cancellationToken).ConfigureAwait(false);
                }

                return tempPath; // ownership transfers to the caller/ProcessDownloadedPostAsync from here
            }
            catch
            {
                // Partial file from a failed/cancelled copy — clean it up ourselves since
                // no return value is escaping to hand cleanup responsibility onward.
                if (File.Exists(tempPath))
                    File.Delete(tempPath);
                throw;
            }
        }
        finally
        {
            downloadPool.Release();
        }
    }

    /// <summary>
    /// Takes a temp file <see cref="DownloadPostAsync"/> already downloaded, decodes/
    /// hashes/resizes it (gated by <paramref name="processingPool"/>), re-checks dedup
    /// under <paramref name="stateLock"/> (see <see cref="DownloadPostAsync"/>'s own doc
    /// comment for why a fresh check is needed here), and either commits a new row or
    /// discards it as a duplicate discovered in the meantime. Always deletes
    /// <paramref name="tempPath"/> before returning, regardless of outcome. Returns
    /// whether a new image was actually appended (for overall-progress bookkeeping) —
    /// any dedup skip or an undecodable download return false.
    ///
    /// <paramref name="reportOverall"/> is invoked from inside the same
    /// <paramref name="stateLock"/> section that appends to <paramref name="writer"/>,
    /// not by the caller afterward: <c>writer.ImageCount</c> only ever grows, but reading
    /// it and reporting it OUTSIDE the lock let two concurrent posts' reports land out of
    /// write-order (post A reads a lower count, post B reads and reports a higher one
    /// first, then A's stale, lower report overwrites it) — the exact same class of bug
    /// as the one <c>progressLock</c> fixes in <see cref="RunSiteTagPhaseAsync"/> for the
    /// per-site rows, just for the one counter that's genuinely global across sites.
    /// </summary>
    private static async Task<bool> ProcessDownloadedPostAsync(
        string tempPath,
        BooruPost post,
        string site,
        string? searchedTag,
        TagVocabulary vocabulary,
        PreprocessedDatasetCacheWriter writer,
        IReadOnlyDictionary<string, string> eligibleTagIdentities,
        int inputSize,
        CrawlWorkingState state,
        SemaphoreSlim stateLock,
        SemaphoreSlim processingPool,
        Action<long, long>? reportOverall,
        long estimatedTotal,
        CancellationToken cancellationToken)
    {
        try
        {
            ulong phash;
            EncodedImage preprocessed;
            await processingPool.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                // Decoded once and reused for both: phash and cache-encoding used to each
                // decode the same file independently, which a profiler showed as two full
                // SKCodec.GetPixels passes (~85% of this method's cost, roughly matched) —
                // pure duplicated work against identical bytes. Gated by processingPool
                // (capped at Environment.ProcessorCount, per site) since posts across a
                // page can now be in this block concurrently; without a cap, a big page
                // would decode/resize dozens of full-resolution images at once and thrash
                // rather than actually go faster.
                using var bitmap = SKBitmap.Decode(tempPath)
                    ?? throw new InvalidDataException($"Could not decode image at '{tempPath}'.");
                phash = PerceptualHash.Compute(bitmap);
                preprocessed = ImagePreprocessing.Encode(bitmap, inputSize);
            }
            catch (InvalidDataException)
            {
                // Corrupt/unsupported download — skip this post entirely rather than
                // recording a broken image or crashing the whole crawl run.
                return false;
            }
            finally
            {
                processingPool.Release();
            }

            await stateLock.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                // Re-check everything: another site's worker (or another post from this
                // same page) may have committed this exact post or a near-duplicate while
                // this one's download/decode was in flight.
                if (state.KnownImages.TryGetValue(post.Md5, out _))
                {
                    var observedTags = MergeDuplicateTags(post, post.Md5, site, searchedTag, vocabulary, eligibleTagIdentities, state);
                    state.PendingAdditionalSources.Add(new PendingAdditionalSource(
                        post.Md5, site, post.PostId, post.FileUrl.ToString(), post.Rating, post.CreatedAt, observedTags, DateTimeOffset.UtcNow));
                    RecordDuplicateIfSearched(state, site, searchedTag);
                    return false;
                }

                var duplicateMd5 = state.HashIndex.FindNear(phash);
                if (duplicateMd5 is not null)
                {
                    // Same artwork, re-encoded by this site — attribute it as another source
                    // of the already-cached (canonical) image rather than appending a
                    // near-identical duplicate under a different md5.
                    var observedTags = MergeDuplicateTags(post, duplicateMd5, site, searchedTag, vocabulary, eligibleTagIdentities, state);
                    state.PendingAdditionalSources.Add(new PendingAdditionalSource(
                        duplicateMd5, site, post.PostId, post.FileUrl.ToString(), post.Rating, post.CreatedAt, observedTags, DateTimeOffset.UtcNow));
                    RecordDuplicateIfSearched(state, site, searchedTag);
                    return false;
                }

                var eligibleTagsOnPost = TagRowMutations.EligibleIdentities(post.Tags, eligibleTagIdentities).Distinct(StringComparer.Ordinal).ToList();
                var tagRows = new List<int>(eligibleTagsOnPost.Count);
                foreach (var tagName in eligibleTagsOnPost)
                {
                    var record = vocabulary.RecordObservation(tagName);
                    tagRows.Add(record.RowIndex);
                }

                writer.Append(preprocessed, tagRows);
                var cacheRowIndex = writer.ImageCount - 1;

                state.KnownImages[post.Md5] = cacheRowIndex;
                state.HashIndex.Add(phash, post.Md5);
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

                reportOverall?.Invoke(writer.ImageCount, Math.Max(estimatedTotal, writer.ImageCount));

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
    /// Credits <paramref name="site"/>'s live, in-memory <see cref="CrawlWorkingState.SiteDuplicateCounts"/>
    /// for a duplicate post found while searching for <paramref name="searchedTag"/> — a
    /// no-op during the negative phase, where <paramref name="searchedTag"/> is null (see
    /// <see cref="RunSiteTagPhaseAsync"/>) since a duplicate there isn't "for" any single
    /// tag. Called alongside every <see cref="MergeDuplicateTags"/> call in
    /// <see cref="DownloadPostAsync"/>/<see cref="ProcessDownloadedPostAsync"/>, always
    /// already holding <c>stateLock</c>. Only ever read for progress display
    /// (<see cref="RunSiteTagPhaseAsync"/>'s <c>currentTagDuplicates</c>) — never a quota
    /// input.
    /// </summary>
    private static void RecordDuplicateIfSearched(CrawlWorkingState state, string site, string? searchedTag)
    {
        if (searchedTag is null)
            return;

        var dupCounts = state.SiteDuplicateCounts[site];
        dupCounts[searchedTag] = dupCounts.GetValueOrDefault(searchedTag) + 1;
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
    /// <c>stateLock</c> — see <see cref="DownloadPostAsync"/>/<see cref="ProcessDownloadedPostAsync"/>.
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
        IReadOnlyDictionary<string, string> eligibleTagIdentities,
        CrawlWorkingState state)
    {
        var cacheRowIndex = state.KnownImages[canonicalMd5];
        var currentTagRows = state.ImageTagRowsByCacheRow[cacheRowIndex];

        var observedTags = TagRowMutations.EligibleIdentities(post.Tags, eligibleTagIdentities).Distinct(StringComparer.Ordinal).ToList();

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
