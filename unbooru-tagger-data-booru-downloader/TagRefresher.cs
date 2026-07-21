using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Crawler;

/// <summary><see cref="TagRefresher.RunAsync"/>'s outcome — how much work happened, and any site that went unavailable partway through (same shape/meaning as <see cref="CrawlResult.FailedSites"/>).</summary>
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
        CancellationToken cancellationToken)
    {
        var sites = clientsBySite.Keys.ToList();

        progress?.ReportPhase("Loading tag survey...");
        var allTags = await db.GetAllSurveyedTagsAsync(cancellationToken).ConfigureAwait(false);
        var eligibleTagSet = allTags.Where(t => TagEligibility.IsEligible(t, minImages)).Select(t => t.Name).ToHashSet(StringComparer.Ordinal);

        progress?.ReportPhase("Loading tag vocabulary...");
        var vocabularyPath = Path.Combine(outputDirectory, "tag_vocabulary.json");
        var vocabularyDeltaPath = Path.Combine(outputDirectory, "tag_vocabulary.delta.jsonl");
        var vocabulary = File.Exists(vocabularyPath)
            ? TagVocabulary.Load(vocabularyPath, vocabularyDeltaPath)
            : TagVocabulary.CreateEmpty();

        progress?.ReportPhase("Opening cache writer...");
        using var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(outputDirectory, inputSize);

        progress?.ReportPhase("Loading dedup index and tag rows...");
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
        var unavailableSites = new Dictionary<string, string>(StringComparer.Ordinal);
        var sourcesChecked = 0;

        async Task CheckpointAsync(string site, long lastPostId, bool done)
        {
            progress?.ReportPhase($"'{site}': checkpointing (tag rows, vocabulary, refresh progress)...");

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
            foreach (var site in sites)
            {
                if (unavailableSites.ContainsKey(site))
                    continue;

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
                        await CheckpointAsync(site, lastPostId, done: true).ConfigureAwait(false);
                        progress?.ReportPhase($"'{site}': up to date — nothing left to refresh.");
                        break;
                    }

                    progress?.ReportPhase($"'{site}': refreshing {batch.Count} post(s) after id {lastPostId}...");

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
                            SiteAvailability.MarkUnavailable(site, ex, sites, unavailableSites, msg => progress?.ReportPhase(msg));
                            siteFailed = true;
                            break;
                        }

                        sourcesChecked++;
                        progress?.ReportOverall(sourcesChecked, sourcesChecked);
                        var observedTags = post?.Tags.Where(eligibleTagSet.Contains).Distinct(StringComparer.Ordinal).ToList() ?? [];

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

                    await CheckpointAsync(site, lastPostId, done: false).ConfigureAwait(false);
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
