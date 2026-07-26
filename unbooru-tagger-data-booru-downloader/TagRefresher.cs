using System.Collections.Concurrent;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Crawler;

/// <summary>
/// <see cref="TagRefresher.RunAsync"/>'s outcome — how much work happened, and any site
/// that went unavailable partway through and was dropped for the rest of the run (see
/// <see cref="SiteAvailability.MarkUnavailable"/>). Unlike <see cref="DatasetCrawler"/>'s
/// per-site workers, a refresh sweep still gives up on a site outright rather than
/// retrying with backoff — resumable via <c>RefreshProgress</c> either way, so a later
/// re-run picks a dropped site back up from where it left off.
/// </summary>
public sealed record RefreshResult(int SourcesChecked, int ImagesChanged, IReadOnlyDictionary<string, string> FailedSites);

/// <summary>
/// Implements the <c>refresh-tags</c> command: re-fetches previously-crawled posts by
/// their stored (site, post id) — not by re-listing tags — to catch tag edits made on
/// the site since <c>crawl</c> last saw them, something a normal crawl only catches by
/// accident (if pagination happens to re-list the same post). Reconciles each affected
/// image's tag set as the union of every known source's <em>current</em> tags, which can
/// both add and remove a tag — unlike <see cref="DatasetCrawler"/>'s own duplicate-merge
/// path, which only ever adds (it can't afford to load every source of an image just to
/// check a mid-crawl duplicate; this command's whole job is exactly that check, one post
/// at a time, so the cost is expected and bounded).
///
/// Never removes a tag while any source's snapshot is still unknown (never captured) —
/// only once <em>every</em> known source of an image has a real snapshot does dropping a
/// tag become safe. This is what makes an old, pre-this-feature dataset safe to run
/// against: nothing shrinks until every source has actually been verified at least once.
///
/// Resumable per site via <c>RefreshProgress</c> (mirrors <see cref="TagProgressState"/>):
/// a normal (non-<c>--reset</c>) run only ever asks for sources past the last one it
/// checked, so re-running it after nothing new has been crawled is a cheap no-op, and
/// running it after a fresh <c>crawl</c> pass added more sources picks up just those.
/// <c>--reset</c> starts a site's sweep over from the beginning, for verifying posts
/// this command itself hasn't touched yet (or re-verifying everything after a change to
/// how reconciliation works).
///
/// <see cref="RunAsync"/>'s <c>onlyTagsAffectingImages</c> parameter is a separate,
/// targeted mode: instead of the full per-site cursor sweep above, it resolves exactly
/// which already-known images currently hold one of the given tags and re-checks only
/// those images' sources — for a scoped correction (e.g. reconciling images stuck on a
/// tag identity a tag-alias merge just orphaned) where the full sweep would be
/// enormously more work than the problem actually requires. It deliberately never reads
/// or writes <c>RefreshProgress</c> (a targeted pass touches an arbitrary subset of post
/// ids, not an ordered walk, so it must neither advance nor reset the full sweep's own
/// resumable cursor) and re-derives its own working set fresh on every run, so simply
/// re-running it after an interruption skips whatever already got reconciled — any image
/// no longer holding one of the target tags just won't be found again.
/// </summary>
public static class TagRefresher
{
    /// <summary>Sources per checkpoint — small on purpose: each one costs a real HTTP request at the site's own rate limit (a few per second), so a checkpoint arrives every few seconds to tens of seconds, never long enough for an interruption to cost much.</summary>
    private const int BatchSize = 50;

    public static async Task<RefreshResult> RunAsync(
        CrawlDatabase db,
        IReadOnlyDictionary<string, IBooruClient> clientsBySite,
        string outputDirectory,
        int inputSize,
        int minImages,
        bool reset,
        CrawlProgressReporter? progress,
        CancellationToken cancellationToken,
        TagExclusionRules? excludedTags = null,
        IReadOnlyDictionary<string, string>? tagAliases = null,
        IReadOnlyCollection<string>? onlyTagsAffectingImages = null)
    {
        var sites = clientsBySite.Keys.ToList();

        // Setup below is shared, one-time work before any site's sweep starts — every
        // site's row briefly shows the same status rather than picking one arbitrarily.
        void ReportSetupPhase(string phase)
        {
            foreach (var siteReporter in progress?.Sites.Values ?? [])
                siteReporter.ReportPhase(phase);
        }

        ReportSetupPhase("Loading tag survey...");
        var allTags = await db.GetAllSurveyedTagsAsync(cancellationToken).ConfigureAwait(false);
        // Keyed by raw (un-prefixed) booru name, not identity — the only form a post's
        // tags ever come back as. See TagRowMutations.BuildEligibleIdentities — tagAliases
        // is what lets this command actually correct an image already mistagged under an
        // aliased-away identity (e.g. head_pat): a fresh re-fetch of that same Gelbooru
        // post still reports the raw head_pat string, and without this it would resolve
        // to nothing at all rather than the merged headpat identity, leaving the old,
        // now-ineligible tag in place forever instead of reconciling it away.
        var eligibleTagIdentities = TagRowMutations.BuildEligibleIdentities(
            allTags.Where(t => TagEligibility.IsEligible(t, minImages, excludedTags)), tagAliases);

        ReportSetupPhase("Loading tag vocabulary...");
        var vocabularyPath = Path.Combine(outputDirectory, "tag_vocabulary.json");
        var vocabularyDeltaPath = Path.Combine(outputDirectory, "tag_vocabulary.delta.jsonl");
        var vocabulary = File.Exists(vocabularyPath)
            ? TagVocabulary.LoadAndCompact(vocabularyPath, vocabularyDeltaPath)
            : TagVocabulary.CreateEmpty();

        ReportSetupPhase("Opening cache writer...");
        using var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(outputDirectory, inputSize);

        ReportSetupPhase("Loading dedup index and tag rows...");
        var existingImages = await db.GetAllImagesAsync(cancellationToken).ConfigureAwait(false);
        CacheConsistency.Validate(existingImages, writer.ImageCount, outputDirectory);
        var knownImages = existingImages.ToDictionary(e => e.Md5, e => e.CacheRowIndex, StringComparer.Ordinal);
        var committedTagRows = writer.ReadCommittedTagRows();
        var imageTagRowsByCacheRow = existingImages.ToDictionary(
            e => e.CacheRowIndex,
            e => new HashSet<int>(committedTagRows[e.CacheRowIndex]));

        var dirtyCacheRows = new HashSet<int>();
        var imagesChanged = new HashSet<int>();
        var refreshedSourcesBuffer = new List<RefreshedSourceTags>();
        var combinedPositiveCountDeltas = new Dictionary<string, int>(StringComparer.Ordinal);
        var unavailableSites = new ConcurrentDictionary<string, string>(StringComparer.Ordinal);
        var sourcesChecked = 0;

        async Task CheckpointAsync(SiteProgressReporter? siteReporter, string site, long lastPostId, bool done)
        {
            siteReporter?.ReportPhase($"'{site}': checkpointing (tag rows, vocabulary, refresh progress)...");

            if (dirtyCacheRows.Count > 0)
            {
                writer.MergeTagRows(dirtyCacheRows.ToDictionary(
                    row => row,
                    IReadOnlyList<int> (row) => imageTagRowsByCacheRow[row].ToList()));
                dirtyCacheRows.Clear();
            }
            vocabulary.SaveDelta(vocabularyDeltaPath);

            await db.ApplyRefreshBatchAsync(refreshedSourcesBuffer, combinedPositiveCountDeltas, site, lastPostId, done, cancellationToken).ConfigureAwait(false);
            refreshedSourcesBuffer.Clear();
            combinedPositiveCountDeltas.Clear();
        }

        try
        {
            if (onlyTagsAffectingImages is { Count: > 0 })
            {
                var targetRowIndices = onlyTagsAffectingImages
                    .Select(tag => vocabulary.TryGet(tag, out var record) ? record.RowIndex : (int?)null)
                    .Where(rowIndex => rowIndex is not null)
                    .Select(rowIndex => rowIndex!.Value)
                    .ToHashSet();

                var md5ByCacheRow = knownImages.ToDictionary(kv => kv.Value, kv => kv.Key);
                var affectedMd5s = imageTagRowsByCacheRow
                    .Where(kv => kv.Value.Overlaps(targetRowIndices))
                    .Select(kv => md5ByCacheRow[kv.Key])
                    .ToList();

                ReportSetupPhase($"Resolving sources for {affectedMd5s.Count} targeted image(s)...");

                var sourcesBySite = new Dictionary<string, List<(long PostId, string Md5)>>(StringComparer.Ordinal);
                foreach (var md5 in affectedMd5s)
                {
                    foreach (var snapshot in await db.GetImageSourceSnapshotsAsync(md5, cancellationToken).ConfigureAwait(false))
                    {
                        if (!clientsBySite.ContainsKey(snapshot.Site))
                            continue; // no client configured for this source's site — can't refetch it this run

                        if (!sourcesBySite.TryGetValue(snapshot.Site, out var list))
                            sourcesBySite[snapshot.Site] = list = [];
                        list.Add((snapshot.PostId, md5));
                    }
                }

                foreach (var site in sites)
                {
                    if (unavailableSites.ContainsKey(site) || !sourcesBySite.TryGetValue(site, out var siteSources))
                        continue;

                    var siteReporter = progress?.Sites.GetValueOrDefault(site);
                    var client = clientsBySite[site];
                    // Snapshot the real cursor once and pass it straight back on every
                    // checkpoint, unchanged — see this class's own doc comment on why a
                    // targeted pass must never touch RefreshProgress for real.
                    var (untouchedLastPostId, untouchedDone) = await db.GetRefreshProgressAsync(site, cancellationToken).ConfigureAwait(false);

                    for (var offset = 0; offset < siteSources.Count; offset += BatchSize)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var batch = siteSources.Skip(offset).Take(BatchSize).ToList();
                        siteReporter?.ReportPhase($"'{site}': refreshing {batch.Count} targeted post(s) ({Math.Min(offset + batch.Count, siteSources.Count)}/{siteSources.Count})...");

                        var siteFailed = false;
                        foreach (var (postId, md5) in batch)
                        {
                            if (!knownImages.TryGetValue(md5, out var cacheRowIndex))
                                continue;

                            BooruPost? post;
                            try
                            {
                                post = await client.GetPostAsync(postId, cancellationToken).ConfigureAwait(false);
                            }
                            catch (Exception ex) when (ex is not OperationCanceledException)
                            {
                                SiteAvailability.MarkUnavailable(site, ex, sites, unavailableSites, msg => siteReporter?.ReportPhase(msg));
                                siteFailed = true;
                                break;
                            }

                            sourcesChecked++;
                            progress?.ReportOverall(sourcesChecked, sourcesChecked);
                            var observedTags = post is null
                                ? []
                                : TagRowMutations.EligibleIdentities(post.Tags, eligibleTagIdentities).Distinct(StringComparer.Ordinal).ToList();

                            if (ReconcileImageTags(vocabulary, md5, site, postId, observedTags, cacheRowIndex, imageTagRowsByCacheRow,
                                    await db.GetImageSourceSnapshotsAsync(md5, cancellationToken).ConfigureAwait(false),
                                    combinedPositiveCountDeltas))
                            {
                                dirtyCacheRows.Add(cacheRowIndex);
                                imagesChanged.Add(cacheRowIndex);
                            }

                            refreshedSourcesBuffer.Add(new RefreshedSourceTags(site, postId, md5, observedTags, DateTimeOffset.UtcNow));
                        }

                        await CheckpointAsync(siteReporter, site, untouchedLastPostId, untouchedDone).ConfigureAwait(false);

                        if (siteFailed)
                            break;
                    }
                }

                if (File.Exists(vocabularyDeltaPath))
                {
                    vocabulary.Save(vocabularyPath);
                    File.Delete(vocabularyDeltaPath);
                }

                return new RefreshResult(sourcesChecked, imagesChanged.Count, unavailableSites);
            }

            foreach (var site in sites)
            {
                if (unavailableSites.ContainsKey(site))
                    continue;

                var siteReporter = progress?.Sites.GetValueOrDefault(site);
                var client = clientsBySite[site];
                var (savedLastPostId, _) = await db.GetRefreshProgressAsync(site, cancellationToken).ConfigureAwait(false);
                var lastPostId = reset ? 0L : savedLastPostId;

                var siteFailed = false;
                while (!siteFailed)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    var batch = await db.GetSourcesBatchAsync(site, lastPostId, BatchSize, cancellationToken).ConfigureAwait(false);
                    if (batch.Count == 0)
                    {
                        await CheckpointAsync(siteReporter, site, lastPostId, done: true).ConfigureAwait(false);
                        siteReporter?.ReportPhase($"'{site}': up to date — nothing left to refresh.");
                        break;
                    }

                    siteReporter?.ReportPhase($"'{site}': refreshing {batch.Count} post(s) after id {lastPostId}...");

                    foreach (var (postId, md5) in batch)
                    {
                        if (!knownImages.TryGetValue(md5, out var cacheRowIndex))
                        {
                            // An ImageSources row should always have a matching Images
                            // row — but don't let a data inconsistency kill the sweep.
                            lastPostId = postId;
                            continue;
                        }

                        BooruPost? post;
                        try
                        {
                            post = await client.GetPostAsync(postId, cancellationToken).ConfigureAwait(false);
                        }
                        catch (Exception ex) when (ex is not OperationCanceledException)
                        {
                            SiteAvailability.MarkUnavailable(site, ex, sites, unavailableSites, msg => siteReporter?.ReportPhase(msg));
                            siteFailed = true;
                            break;
                        }

                        sourcesChecked++;
                        progress?.ReportOverall(sourcesChecked, sourcesChecked);
                        var observedTags = post is null
                            ? []
                            : TagRowMutations.EligibleIdentities(post.Tags, eligibleTagIdentities).Distinct(StringComparer.Ordinal).ToList();

                        if (ReconcileImageTags(vocabulary, md5, site, postId, observedTags, cacheRowIndex, imageTagRowsByCacheRow,
                                await db.GetImageSourceSnapshotsAsync(md5, cancellationToken).ConfigureAwait(false),
                                combinedPositiveCountDeltas))
                        {
                            dirtyCacheRows.Add(cacheRowIndex);
                            imagesChanged.Add(cacheRowIndex);
                        }

                        // Always the real (possibly empty) observed list, never null —
                        // a deleted post is a *known* zero-tags state, not "unknown";
                        // see RefreshedSourceTags's own doc comment for why that
                        // distinction is what makes removal ever actually trigger.
                        refreshedSourcesBuffer.Add(new RefreshedSourceTags(site, postId, md5, observedTags, DateTimeOffset.UtcNow));
                        lastPostId = postId;
                    }

                    await CheckpointAsync(siteReporter, site, lastPostId, done: false).ConfigureAwait(false);
                }
            }

            // Fold the delta back into the base snapshot once at the end, the same
            // cleanup DatasetCrawler does after its own run — a refresh pass typically
            // touches far fewer tags than a full crawl, so there's no need for
            // DatasetCrawler's periodic mid-run compaction, just this final one.
            if (File.Exists(vocabularyDeltaPath))
            {
                vocabulary.Save(vocabularyPath);
                File.Delete(vocabularyDeltaPath);
            }

            return new RefreshResult(sourcesChecked, imagesChanged.Count, unavailableSites);
        }
        finally
        {
            progress?.Dispose();
        }
    }

    /// <summary>
    /// Reconciles one image's live tag-row set against every known source's tags after
    /// one of those sources (<paramref name="site"/>, <paramref name="postId"/>) was just
    /// refetched. Adding is always safe. Removing a tag only happens when every other
    /// source of this image has a real (non-null) snapshot — otherwise an unverified
    /// sibling source might be the only thing still asserting it, and dropping it would
    /// be a real, silent loss of a training label. Returns whether anything changed.
    /// </summary>
    private static bool ReconcileImageTags(
        TagVocabulary vocabulary,
        string md5,
        string site,
        long postId,
        IReadOnlyList<string> observedTags,
        int cacheRowIndex,
        Dictionary<int, HashSet<int>> imageTagRowsByCacheRow,
        IReadOnlyList<ImageSourceSnapshot> allSourceSnapshots,
        Dictionary<string, int> combinedPositiveCountDeltas)
    {
        var otherSnapshots = allSourceSnapshots.Where(s => !(s.Site == site && s.PostId == postId)).ToList();
        var hasUnknownOtherSource = otherSnapshots.Any(s => s.Tags is null);
        var reconciledTagNames = otherSnapshots
            .Where(s => s.Tags is not null)
            .SelectMany(s => s.Tags!)
            .Concat(observedTags)
            .ToHashSet(StringComparer.Ordinal);

        var currentTagRows = imageTagRowsByCacheRow[cacheRowIndex];
        var changed = false;

        foreach (var tagName in reconciledTagNames)
        {
            if (!TagRowMutations.TryAddTagToImage(vocabulary, tagName, currentTagRows))
                continue;

            combinedPositiveCountDeltas[tagName] = combinedPositiveCountDeltas.GetValueOrDefault(tagName) + 1;
            changed = true;
        }

        if (hasUnknownOtherSource)
            return changed;

        var reconciledRowIndices = reconciledTagNames
            .Select(name => vocabulary.TryGet(name, out var record) ? record.RowIndex : (int?)null)
            .Where(rowIndex => rowIndex is not null)
            .Select(rowIndex => rowIndex!.Value)
            .ToHashSet();

        foreach (var rowIndex in currentTagRows.Except(reconciledRowIndices).ToList())
        {
            var tagName = vocabulary.GetByRowIndex(rowIndex).Tag;
            TagRowMutations.RemoveTagFromImage(vocabulary, rowIndex, currentTagRows);
            combinedPositiveCountDeltas[tagName] = combinedPositiveCountDeltas.GetValueOrDefault(tagName) - 1;
            changed = true;
        }

        return changed;
    }
}
