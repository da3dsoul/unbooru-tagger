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

/// <summary>
/// <see cref="DatasetCrawler.RunAsync"/>'s outcome: the usual per-tag shortfalls, plus
/// any site that went unavailable partway through and was dropped for the rest of the
/// run (see <see cref="DatasetCrawler.RunTagPhaseAsync"/>). Empty when every configured
/// site stayed healthy the whole run. Per-(tag,site,phase) progress for a dropped site
/// is left exactly where its last successful page left it — not marked done — so the
/// next run picks that site back up automatically instead of needing anything special
/// to "resume" it.
/// </summary>
public sealed record CrawlResult(IReadOnlyList<TagShortfall> Shortfalls, IReadOnlyDictionary<string, string> FailedSites);

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
/// </summary>
internal sealed class CrawlWorkingState
{
    public required Dictionary<string, int> KnownImages { get; init; }
    public required List<(ulong Hash, string Md5)> HashIndex { get; init; }
    public required Dictionary<string, int> CombinedPositiveCounts { get; init; }

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
/// both sites, followed by an automatic negative top-up pass. Downloads never persist
/// as a raw-file corpus — each new (post-dedup) image goes to a <c>.tmp</c> scratch file
/// just long enough to decode/normalize via <see cref="ImagePreprocessing.LoadAndNormalize(string, int)"/>
/// and append to the same <see cref="PreprocessedDatasetCacheWriter"/>/<see cref="TagVocabulary"/>
/// format <c>build-large-cache</c> produces, so <c>--output-dir</c> is immediately a
/// trainable dataset directory, not a raw dump needing a separate import step.
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
    /// be treated as the same cross-site re-encode. Kept low/"strict" on purpose: this
    /// index only needs to catch the same source file re-compressed/resized by a
    /// different site, which typically differs by a handful of bits at most — a looser
    /// threshold risks collapsing two genuinely different (but visually similar) images
    /// into one, silently dropping a real training example instead of a true duplicate.
    /// </summary>
    private const int MaxHammingDistance = 6;

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
        CancellationToken cancellationToken)
    {
        var sites = clientsBySite.Keys.ToList();
        var requestsBySite = new Dictionary<string, int>(StringComparer.Ordinal);

        progress?.ReportPhase("Loading tag survey...");
        var allTags = await db.GetAllSurveyedTagsAsync(cancellationToken).ConfigureAwait(false);
        var eligibleTags = CrawlScheduling.RarestFirst(allTags.Where(t => TagEligibility.IsEligible(t, minImages))).ToList();
        var estimatedTotal = TagEligibility.EstimateImageSlots(eligibleTags, minImages, maxImages);

        progress?.ReportPhase("Loading tag vocabulary...");
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

        progress?.ReportPhase("Opening cache writer...");
        using var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(outputDirectory, inputSize);

        // Seeded once from durable state (accurate as of the last successful checkpoint,
        // exactly in step with the cache file writer above thanks to its own
        // truncate-to-last-flush resume logic), then kept live in memory from here on —
        // see CrawlWorkingState's own doc comment for why the durable copies deliberately
        // lag behind these during a run.
        progress?.ReportPhase("Loading dedup index and tag counters...");
        var existingImages = await db.GetAllImagesAsync(cancellationToken).ConfigureAwait(false);
        CacheConsistency.Validate(existingImages, writer.ImageCount, outputDirectory);
        var combinedPositiveCounts = await db.GetAllCombinedPositiveCountsAsync(cancellationToken).ConfigureAwait(false);
        var committedTagRows = writer.ReadCommittedTagRows();
        var state = new CrawlWorkingState
        {
            KnownImages = existingImages.ToDictionary(e => e.Md5, e => e.CacheRowIndex, StringComparer.Ordinal),
            HashIndex = existingImages.Select(e => (e.PHash, e.Md5)).ToList(),
            CombinedPositiveCounts = new Dictionary<string, int>(combinedPositiveCounts, StringComparer.Ordinal),
            ImageTagRowsByCacheRow = existingImages.ToDictionary(
                e => e.CacheRowIndex,
                e => new HashSet<int>(committedTagRows[e.CacheRowIndex])),
        };

        var pageCounter = 0;

        async Task CheckpointAsync()
        {
            progress?.ReportPhase("Checkpointing (cache, vocabulary, dedup index)...");

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
                await db.CommitPendingImagesAsync(state.PendingNewImages, state.PendingAdditionalSources, state.PendingMergedTagCounts, cancellationToken).ConfigureAwait(false);
                state.PendingNewImages.Clear();
                state.PendingAdditionalSources.Clear();
                state.PendingMergedTagCounts.Clear();
            }

            pageCounter++;
            if (pageCounter % vocabCompactIntervalPages == 0)
            {
                progress?.ReportPhase("Compacting vocabulary...");
                vocabulary.Save(vocabularyPath);
                File.Delete(vocabularyDeltaPath);
            }
        }

        var shortfalls = new List<TagShortfall>();

        // Site -> failure reason. Shared across every tag/phase for the life of this
        // run: once a site fails it stays dropped rather than being retried (and
        // exhausting a full backoff budget) on every subsequent tag — see
        // RunTagPhaseAsync/SiteAvailability.MarkUnavailable. Never persisted; a fresh
        // process starts with every site available again, so a dropped site is
        // automatically retried on the next run from wherever its per-(tag,site,phase)
        // cursor left off.
        var unavailableSites = new Dictionary<string, string>(StringComparer.Ordinal);

        try
        {
            progress?.ReportOverall(writer.ImageCount, Math.Max(estimatedTotal, writer.ImageCount));
            progress?.ReportTagsCompleted("Positive pass", 0, eligibleTags.Count);

            var tagIndex = 0;
            foreach (var tag in eligibleTags)
            {
                tagIndex++;
                cancellationToken.ThrowIfCancellationRequested();

                progress?.ReportPhase($"Positive crawl: tag '{tag.Name}' ({tagIndex}/{eligibleTags.Count} eligible, target {maxImages})");

                // The tag's own realistic ceiling, not the raw --max-images target: many
                // (especially rarer) eligible tags simply don't have --max-images posts
                // available on either site at all, in which case the tag's own progress
                // row should read "done at its real max" rather than stall short of a
                // target it could never reach — this is the same min(maxImages, count)
                // TagEligibility.EstimateImageSlots already uses per tag.
                var tagCeiling = Math.Min(maxImages, tag.BestCount);

                await RunTagPhaseAsync(
                    PositivePhase,
                    tag.Name,
                    tag.Name,
                    () => CrawlQuota.ShouldContinueFetching(state.CombinedPositiveCount(tag.Name), maxImages),
                    sites,
                    clientsBySite,
                    requestsBySite,
                    db,
                    vocabulary,
                    writer,
                    eligibleTagSet,
                    tempDir,
                    inputSize,
                    downloadClient,
                    state,
                    progress,
                    estimatedTotal,
                    CheckpointAsync,
                    unavailableSites,
                    () => state.CombinedPositiveCount(tag.Name),
                    tagCeiling,
                    cancellationToken).ConfigureAwait(false);

                var finalCount = state.CombinedPositiveCount(tag.Name);
                if (finalCount < maxImages)
                    shortfalls.Add(new TagShortfall(tag.Name, finalCount, maxImages));

                progress?.ReportTagsCompleted("Positive pass", tagIndex, eligibleTags.Count);
            }

            progress?.ReportPhase($"Entering negative top-up phase — target {negativeTarget} non-tagged images per eligible tag");

            tagIndex = 0;
            progress?.ReportTagsCompleted("Negative pass", 0, eligibleTags.Count);
            foreach (var tag in eligibleTags)
            {
                tagIndex++;
                cancellationToken.ThrowIfCancellationRequested();

                var negativeQuery = $"-{tag.Name}";
                progress?.ReportPhase($"Negative top-up: tag '{tag.Name}' ({tagIndex}/{eligibleTags.Count})");

                await RunTagPhaseAsync(
                    NegativePhase,
                    tag.Name,
                    negativeQuery,
                    () => CrawlQuota.NegativeShortfall(writer.ImageCount, state.CombinedPositiveCount(tag.Name), negativeTarget) > 0,
                    sites,
                    clientsBySite,
                    requestsBySite,
                    db,
                    vocabulary,
                    writer,
                    eligibleTagSet,
                    tempDir,
                    inputSize,
                    downloadClient,
                    state,
                    progress,
                    estimatedTotal,
                    CheckpointAsync,
                    unavailableSites,
                    () => writer.ImageCount - state.CombinedPositiveCount(tag.Name),
                    negativeTarget,
                    cancellationToken).ConfigureAwait(false);

                progress?.ReportTagsCompleted("Negative pass", tagIndex, eligibleTags.Count);
            }

            progress?.ReportPhase("Compacting vocabulary...");
            vocabulary.Save(vocabularyPath);
            if (File.Exists(vocabularyDeltaPath))
                File.Delete(vocabularyDeltaPath);
            writer.Flush();

            progress?.ReportPhase("Done.");

            return new CrawlResult(shortfalls, unavailableSites);
        }
        finally
        {
            progress?.Dispose();
        }
    }

    /// <summary>
    /// Shared loop for both the positive crawl and the negative top-up: repeatedly picks
    /// the least-loaded site that hasn't exhausted this tag/phase's pagination and hasn't
    /// gone unavailable this run, fetches one page, processes each post (dedup-skip or
    /// download+append), checkpoints, and only then persists per-(tag,site,phase) cursor
    /// progress — until either <paramref name="shouldContinue"/> says the target is met
    /// or every remaining site is exhausted/unavailable for this tag/phase.
    ///
    /// A page fetch or post download that fails outright (not the transient blips
    /// <see cref="TransientHttpRetry"/> already retries — this is what's left once that's
    /// exhausted) marks its site unavailable in <paramref name="unavailableSites"/> and
    /// moves on to whatever site is left, rather than taking the whole run down; see
    /// <see cref="SiteAvailability.MarkUnavailable"/> for what happens once none are left.
    /// </summary>
    private static async Task RunTagPhaseAsync(
        string phase,
        string progressTagName,
        string tagQuery,
        Func<bool> shouldContinue,
        IReadOnlyList<string> sites,
        IReadOnlyDictionary<string, IBooruClient> clientsBySite,
        Dictionary<string, int> requestsBySite,
        CrawlDatabase db,
        TagVocabulary vocabulary,
        PreprocessedDatasetCacheWriter writer,
        HashSet<string> eligibleTagSet,
        string tempDir,
        int inputSize,
        HttpClient downloadClient,
        CrawlWorkingState state,
        CrawlProgressReporter? progress,
        long estimatedTotal,
        Func<Task> checkpointAsync,
        Dictionary<string, string> unavailableSites,
        Func<int> currentTagProgress,
        int tagProgressTarget,
        CancellationToken cancellationToken)
    {
        progress?.ReportTagProgress(progressTagName, currentTagProgress(), tagProgressTarget);

        while (shouldContinue())
        {
            cancellationToken.ThrowIfCancellationRequested();

            var candidateSites = new List<string>();
            foreach (var site in sites)
            {
                if (unavailableSites.ContainsKey(site))
                    continue;

                var siteProgress = await db.GetTagProgressAsync(progressTagName, site, phase, cancellationToken).ConfigureAwait(false);
                if (!siteProgress.Done)
                    candidateSites.Add(site);
            }

            if (candidateSites.Count == 0)
                break; // remaining sites exhausted (or all unavailable) without meeting the target for this tag/phase

            var chosenSite = CrawlScheduling.PickLeastLoadedSite(
                candidateSites.ToDictionary(s => s, s => requestsBySite.GetValueOrDefault(s, 0)));

            var tagProgress = await db.GetTagProgressAsync(progressTagName, chosenSite, phase, cancellationToken).ConfigureAwait(false);
            var client = clientsBySite[chosenSite];

            BooruPostPage page;
            try
            {
                progress?.ReportPhase($"{phase} crawl: tag '{progressTagName}', fetching page from {chosenSite}...");
                page = await client.ListPostsAsync(tagQuery, tagProgress.Cursor, cancellationToken).ConfigureAwait(false);
                requestsBySite[chosenSite] = requestsBySite.GetValueOrDefault(chosenSite, 0) + 1;
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                SiteAvailability.MarkUnavailable(chosenSite, ex, sites, unavailableSites, msg => progress?.ReportPhase(msg));
                continue;
            }

            var pageIncomplete = false;
            progress?.ReportPhase($"{phase} crawl: tag '{progressTagName}', processing {page.Posts.Count} posts from {chosenSite}...");
            for (var i = 0; i < page.Posts.Count; i++)
            {
                if (!shouldContinue())
                    break;

                progress?.ReportPhaseProgress(i, page.Posts.Count);

                bool appended;
                try
                {
                    appended = await ProcessPostAsync(
                        page.Posts[i], chosenSite, client.RateLimiter, vocabulary, writer, eligibleTagSet, tempDir, inputSize, downloadClient, state, cancellationToken).ConfigureAwait(false);
                }
                catch (Exception ex) when (ex is not OperationCanceledException)
                {
                    // The rest of this page is presumed unreachable too (same dead
                    // site/CDN) — stop processing it rather than retrying each
                    // remaining post individually against a site that's already down.
                    SiteAvailability.MarkUnavailable(chosenSite, ex, sites, unavailableSites, msg => progress?.ReportPhase(msg));
                    pageIncomplete = true;
                    break;
                }

                if (appended)
                    progress?.ReportOverall(writer.ImageCount, Math.Max(estimatedTotal, writer.ImageCount));

                // Unconditional, not just on a fresh append: a dedup-matched duplicate
                // can still credit this tag via ProcessPostAsync's merge path (a known
                // image gaining a tag it didn't have before) without appended being
                // true — gating this on appended alone left the bar showing a stale
                // starting count through however many merge-only credits happened
                // since, which is exactly what made it look frozen at 0 on a page full
                // of already-known images.
                progress?.ReportTagProgress(progressTagName, currentTagProgress(), tagProgressTarget);
            }
            progress?.ReportPhaseProgress(page.Posts.Count, page.Posts.Count);

            // Checkpoint (durably commit this page's work) before advancing the cursor
            // past it — see this file's class-level doc comment on why that ordering is
            // what makes a crash mid-run recoverable instead of silently lossy. Applies
            // even when the page died partway through: whatever posts did get appended
            // before the failure are durable either way.
            await checkpointAsync().ConfigureAwait(false);

            if (pageIncomplete)
                continue; // don't advance the cursor past a page this site didn't finish serving

            var done = page.NextCursor is null;
            await db.SaveTagProgressAsync(
                progressTagName, chosenSite, phase,
                new TagProgressState(page.NextCursor, tagProgress.PostsFetched + page.Posts.Count, done),
                cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Dedup-skips a known image (exact md5 or perceptual near-duplicate — buffers only
    /// an additional-source record), or downloads+normalizes+appends a new one (buffers
    /// a full pending image record). Nothing here talks to <c>crawl.sqlite</c> directly —
    /// everything durable is deferred to the caller's next checkpoint. Returns whether a
    /// new image was actually appended (for overall-progress bookkeeping) — any dedup
    /// skip or an undecodable download return false.
    /// </summary>
    private static async Task<bool> ProcessPostAsync(
        BooruPost post,
        string site,
        IRateLimiter rateLimiter,
        TagVocabulary vocabulary,
        PreprocessedDatasetCacheWriter writer,
        HashSet<string> eligibleTagSet,
        string tempDir,
        int inputSize,
        HttpClient downloadClient,
        CrawlWorkingState state,
        CancellationToken cancellationToken)
    {
        if (state.KnownImages.TryGetValue(post.Md5, out _))
        {
            var observedTags = MergeDuplicateTags(post, post.Md5, vocabulary, eligibleTagSet, state);
            state.PendingAdditionalSources.Add(new PendingAdditionalSource(
                post.Md5, site, post.PostId, post.FileUrl.ToString(), post.Rating, post.CreatedAt, observedTags, DateTimeOffset.UtcNow));
            return false;
        }

        var tempPath = Path.Combine(tempDir, $"{Guid.NewGuid():N}.img");
        try
        {
            // Raw image bytes are typically served from the same site/CDN as the JSON
            // API, so reuse its rate limiter here too — this used to be a completely
            // unthrottled GetStreamAsync, which fired a whole page's worth of downloads
            // back-to-back with no pacing at all and no retry, so a single 429 from the
            // CDN crashed the entire run instead of just backing off.
            using var response = await TransientHttpRetry.SendWithRetryAsync(
                async () =>
                {
                    await rateLimiter.WaitAsync(cancellationToken).ConfigureAwait(false);
                    return await downloadClient.GetAsync(post.FileUrl, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
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

            var duplicate = state.HashIndex.FirstOrDefault(entry => PerceptualHash.HammingDistance(entry.Hash, phash) <= MaxHammingDistance);
            if (duplicate.Md5 is not null)
            {
                // Same artwork, re-encoded by this site — attribute it as another source
                // of the already-cached (canonical) image rather than appending a
                // near-identical duplicate under a different md5.
                var observedTags = MergeDuplicateTags(post, duplicate.Md5, vocabulary, eligibleTagSet, state);
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
            foreach (var tagName in eligibleTagsOnPost)
                state.CombinedPositiveCounts[tagName] = state.CombinedPositiveCount(tagName) + 1;

            state.PendingNewImages.Add(new PendingNewImage(
                post.Md5, cacheRowIndex, post.Width, post.Height, DateTimeOffset.UtcNow,
                site, post.PostId, post.FileUrl.ToString(), post.Rating, post.CreatedAt,
                eligibleTagsOnPost, phash));

            return true;
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
    /// case — most cross-site duplicates agree).
    /// Returns this post's own eligible tags (not just the newly-added ones) so the
    /// caller can record them as this source's fresh snapshot — the ingredient
    /// <c>refresh-tags</c> later reconciles across every source of an image, including
    /// possible removal (see <see cref="TagRefresher"/>); this additive merge itself
    /// never removes anything, since it can't afford to check every other source mid-crawl.
    /// </summary>
    private static IReadOnlyList<string> MergeDuplicateTags(
        BooruPost post,
        string canonicalMd5,
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

        if (newlyAddedTags is not null)
        {
            state.DirtyCacheRows.Add(cacheRowIndex);
            foreach (var tagName in newlyAddedTags)
            {
                state.PendingMergedTagCounts.Add(tagName);
                state.CombinedPositiveCounts[tagName] = state.CombinedPositiveCount(tagName) + 1;
            }
        }

        return observedTags;
    }
}
