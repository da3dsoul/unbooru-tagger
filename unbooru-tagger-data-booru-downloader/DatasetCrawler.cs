using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Crawler;

/// <summary>Up-front, pre-dedup estimate of what a crawl will cost — printed by <c>survey-tags</c> and recomputed as the first step of <c>crawl</c>.</summary>
public sealed record CrawlEstimate(
    int EligibleTagCount,
    long EstimatedImageSlots,
    long EstimatedRequests,
    TimeSpan EstimatedWallClockTime);

/// <summary>
/// Implements the <c>crawl</c> command: a rarest-eligible-tag-first positive pass across
/// both sites, followed by an automatic negative top-up pass. Downloads never persist
/// as a raw-file corpus — each new (post-dedup) image goes to a <c>.tmp</c> scratch file
/// just long enough to decode/normalize via <see cref="ImagePreprocessing.LoadAndNormalize(string, int)"/>
/// and append to the same <see cref="PreprocessedDatasetCacheWriter"/>/<see cref="TagVocabulary"/>
/// format <c>build-large-cache</c> produces, so <c>--output-dir</c> is immediately a
/// trainable dataset directory, not a raw dump needing a separate import step.
/// </summary>
public static class DatasetCrawler
{
    private const string PositivePhase = "positive";
    private const string NegativePhase = "negative";

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

    public static async Task RunAsync(
        CrawlDatabase db,
        IReadOnlyDictionary<string, IBooruClient> clientsBySite,
        HttpClient downloadClient,
        string outputDirectory,
        int inputSize,
        int minImages,
        int maxImages,
        int negativeTarget,
        int checkpointInterval,
        CrawlProgressReporter? progress,
        CancellationToken cancellationToken)
    {
        var sites = clientsBySite.Keys.ToList();
        var requestsBySite = new Dictionary<string, int>(StringComparer.Ordinal);

        var allTags = await db.GetAllSurveyedTagsAsync(cancellationToken).ConfigureAwait(false);
        var eligibleTags = CrawlScheduling.RarestFirst(allTags.Where(t => TagEligibility.IsEligible(t, minImages))).ToList();
        var estimatedTotal = TagEligibility.EstimateImageSlots(eligibleTags, minImages, maxImages);

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
            File.Delete(leftover); // safe: nothing durable is recorded until after a successful cache Append

        using var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(outputDirectory, inputSize);
        var appendedSinceCheckpoint = 0;

        void Checkpoint(int pageNumber)
        {
            vocabulary.SaveDelta(vocabularyDeltaPath);
            writer.Flush();
            if (pageNumber % checkpointInterval == 0)
            {
                vocabulary.Save(vocabularyPath);
                File.Delete(vocabularyDeltaPath);
            }
        }

        try
        {
            progress?.ReportOverall(writer.ImageCount, Math.Max(estimatedTotal, writer.ImageCount));

            var tagIndex = 0;
            foreach (var tag in eligibleTags)
            {
                tagIndex++;
                cancellationToken.ThrowIfCancellationRequested();

                progress?.ReportPhase($"Positive crawl: tag '{tag.Name}' ({tagIndex}/{eligibleTags.Count} eligible, target {maxImages})");

                await RunTagPhaseAsync(
                    PositivePhase,
                    tag.Name,
                    tag.Name,
                    async () => CrawlQuota.ShouldContinueFetching(await db.GetCombinedPositiveCountAsync(tag.Name, cancellationToken).ConfigureAwait(false), maxImages),
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
                    progress,
                    estimatedTotal,
                    () =>
                    {
                        appendedSinceCheckpoint++;
                        if (appendedSinceCheckpoint >= checkpointInterval)
                        {
                            appendedSinceCheckpoint = 0;
                            Checkpoint(writer.ImageCount);
                        }
                    },
                    cancellationToken).ConfigureAwait(false);
            }

            progress?.ReportPhase($"Entering negative top-up phase — target {negativeTarget} non-tagged images per eligible tag");

            tagIndex = 0;
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
                    async () =>
                    {
                        var total = await db.GetTotalImageCountAsync(cancellationToken).ConfigureAwait(false);
                        var positive = await db.GetCombinedPositiveCountAsync(tag.Name, cancellationToken).ConfigureAwait(false);
                        return CrawlQuota.NegativeShortfall(total, positive, negativeTarget) > 0;
                    },
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
                    progress,
                    estimatedTotal,
                    () =>
                    {
                        appendedSinceCheckpoint++;
                        if (appendedSinceCheckpoint >= checkpointInterval)
                        {
                            appendedSinceCheckpoint = 0;
                            Checkpoint(writer.ImageCount);
                        }
                    },
                    cancellationToken).ConfigureAwait(false);
            }

            progress?.ReportPhase("Done.");
            vocabulary.Save(vocabularyPath);
            if (File.Exists(vocabularyDeltaPath))
                File.Delete(vocabularyDeltaPath);
            writer.Flush();
        }
        finally
        {
            progress?.Dispose();
        }
    }

    /// <summary>
    /// Shared loop for both the positive crawl and the negative top-up: repeatedly picks
    /// the least-loaded site that hasn't exhausted this tag/phase's pagination, fetches
    /// one page, processes each post (dedup-skip or download+append), and persists
    /// per-(tag,site,phase) cursor progress after every page — until either
    /// <paramref name="shouldContinue"/> says the target is met or both sites are
    /// exhausted for this tag/phase.
    /// </summary>
    private static async Task RunTagPhaseAsync(
        string phase,
        string progressTagName,
        string tagQuery,
        Func<Task<bool>> shouldContinue,
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
        CrawlProgressReporter? progress,
        long estimatedTotal,
        Action onImageAppended,
        CancellationToken cancellationToken)
    {
        while (await shouldContinue().ConfigureAwait(false))
        {
            cancellationToken.ThrowIfCancellationRequested();

            var candidateSites = new List<string>();
            foreach (var site in sites)
            {
                var siteProgress = await db.GetTagProgressAsync(progressTagName, site, phase, cancellationToken).ConfigureAwait(false);
                if (!siteProgress.Done)
                    candidateSites.Add(site);
            }

            if (candidateSites.Count == 0)
                break; // both sites exhausted without meeting the target for this tag/phase

            var chosenSite = CrawlScheduling.PickLeastLoadedSite(
                candidateSites.ToDictionary(s => s, s => requestsBySite.GetValueOrDefault(s, 0)));

            var tagProgress = await db.GetTagProgressAsync(progressTagName, chosenSite, phase, cancellationToken).ConfigureAwait(false);
            var client = clientsBySite[chosenSite];
            var page = await client.ListPostsAsync(tagQuery, tagProgress.Cursor, cancellationToken).ConfigureAwait(false);
            requestsBySite[chosenSite] = requestsBySite.GetValueOrDefault(chosenSite, 0) + 1;

            foreach (var post in page.Posts)
            {
                if (!await shouldContinue().ConfigureAwait(false))
                    break;

                var appended = await ProcessPostAsync(
                    post, chosenSite, db, vocabulary, writer, eligibleTagSet, tempDir, inputSize, downloadClient, cancellationToken).ConfigureAwait(false);

                if (appended)
                {
                    onImageAppended();
                    progress?.ReportOverall(writer.ImageCount, Math.Max(estimatedTotal, writer.ImageCount));
                }
            }

            var done = page.NextCursor is null;
            await db.SaveTagProgressAsync(
                progressTagName, chosenSite, phase,
                new TagProgressState(page.NextCursor, tagProgress.PostsFetched + page.Posts.Count, done),
                cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Dedup-skips a known md5 (recording only the additional source if this is a new
    /// site/post for an already-cached image), or downloads+normalizes+appends a new
    /// one. Returns whether a new image was actually appended (for progress/checkpoint
    /// bookkeeping) — a dedup skip or an undecodable download both return false.
    /// </summary>
    private static async Task<bool> ProcessPostAsync(
        BooruPost post,
        string site,
        CrawlDatabase db,
        TagVocabulary vocabulary,
        PreprocessedDatasetCacheWriter writer,
        HashSet<string> eligibleTagSet,
        string tempDir,
        int inputSize,
        HttpClient downloadClient,
        CancellationToken cancellationToken)
    {
        var existingRow = await db.FindCacheRowIndexAsync(post.Md5, cancellationToken).ConfigureAwait(false);
        if (existingRow is not null)
        {
            await db.RecordAdditionalSourceAsync(post.Md5, site, post.PostId, post.FileUrl.ToString(), post.Rating, post.CreatedAt, cancellationToken).ConfigureAwait(false);
            return false;
        }

        var tempPath = Path.Combine(tempDir, $"{Guid.NewGuid():N}.img");
        try
        {
            await using (var fileStream = File.Create(tempPath))
            await using (var responseStream = await downloadClient.GetStreamAsync(post.FileUrl, cancellationToken).ConfigureAwait(false))
            {
                await responseStream.CopyToAsync(fileStream, cancellationToken).ConfigureAwait(false);
            }

            float[] pixels;
            try
            {
                pixels = ImagePreprocessing.LoadAndNormalize(tempPath, inputSize);
            }
            catch (InvalidDataException)
            {
                // Corrupt/unsupported download — skip this post entirely rather than
                // recording a broken image or crashing the whole crawl run.
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

            await db.RecordNewImageAsync(
                post.Md5, cacheRowIndex, post.Width, post.Height, DateTimeOffset.UtcNow,
                site, post.PostId, post.FileUrl.ToString(), post.Rating, post.CreatedAt,
                eligibleTagsOnPost, cancellationToken).ConfigureAwait(false);

            return true;
        }
        finally
        {
            if (File.Exists(tempPath))
                File.Delete(tempPath);
        }
    }
}
