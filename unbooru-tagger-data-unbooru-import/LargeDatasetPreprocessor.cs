using Microsoft.Data.SqlClient;
using Microsoft.EntityFrameworkCore;
using unbooru.Abstractions.Poco;
using unbooru.Core;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Data;

/// <summary>
/// Large-mode dataset builder: streams the full (or a large filtered) corpus from
/// unbooru and preprocesses each image exactly once — decode, resize, normalize —
/// directly into a <see cref="PreprocessedDatasetCache"/>, instead of leaving
/// Training to re-decode the same images on every epoch. This is the
/// "maximum speed" mode; "maximum accuracy" comes from pulling the full corpus
/// rather than a capped sample. When a cap is given, selection is tag-coverage-aware
/// (see <see cref="TagCoverageSampleSelector"/>) rather than an arbitrary ImageId-ordered
/// prefix, so a capped cache still gives every known tag a fair shot at training —
/// CLAUDE.md's oversampling batch sampler can only rebalance gradients for tags that
/// actually made it into the cache in the first place.
///
/// Pulls the corpus in keyset-paginated batches (by <c>ImageId</c>) rather than one
/// giant streaming query: a multi-million-image run held open on a single DB
/// connection for hours is fragile against any transient network blip, and when a
/// split query with large included collections (image blobs) drops mid-stream, EF
/// Core can't resume it — the whole run dies. Paginating bounds a hiccup's blast
/// radius to one page, and persisting progress after every page (cache + vocabulary
/// + <see cref="LargeCacheResumeState"/>) means re-running the same command resumes
/// right where it left off instead of restarting from scratch. New tag rows and
/// image-count updates are checkpointed via <see cref="TagVocabulary.SaveDelta"/>
/// (append-only) rather than re-serializing the whole vocabulary every page, so a
/// run's per-page cost doesn't grow with vocabulary size as it scales toward
/// CLAUDE.md's long-tail vocabulary.
/// The delta is periodically folded back into a fresh <c>tag_vocabulary.json</c>
/// snapshot (every <c>vocabCompactionIntervalPages</c> pages, not only once the run
/// finishes) so anything reading that file directly sees reasonably current state.
/// Tags below <c>minTagImageCount</c> images across the whole corpus (CLAUDE.md's
/// minimum-image threshold) never get a vocabulary row or a tag-row entry at all —
/// computed once up front from a grouped count over the tag-link table so it applies
/// uniformly whether or not <c>maxImages</c> caps the run.
///
/// Within a page, decode/resize/normalize (CPU-bound, independent per image) runs in
/// parallel across cores; vocabulary bookkeeping and cache writes are cheap by
/// comparison and stay sequential to avoid needing to synchronize either one.
/// </summary>
/// <summary>Cumulative images preprocessed so far, and the total this run expects to reach — lets a caller render a real percentage/ETA instead of a bare counter.</summary>
public readonly record struct ImageBuildProgress(int Processed, int Total);

/// <summary>
/// Progress sink for <see cref="LargeDatasetPreprocessor.BuildAsync"/>. A single flat
/// percentage isn't a useful "is this actually moving" signal at multi-million-image
/// scale — the overall number can sit still for a long time between visible ticks.
/// Split into overall corpus progress and a phase channel instead. The phase channel
/// covers both a text label for the parts of a page that aren't incrementally
/// measurable (a DB round trip, a checkpoint flush) or that happen once during setup,
/// and a real fraction for the parts that are — the blob fetch (split across several
/// concurrent connections; each one finishing is a concrete, countable step — see
/// <c>FetchBlobsParallel</c>) and the per-page cache write (one image at a time; a
/// dedicated "current page" row added nothing beyond what this fraction already shows,
/// since it only ever moved during this same step). Phase text is caller-rendered
/// (e.g. a live progress row's description), so implementations should bound its length
/// and escape any embedded external text (exception messages, tag strings) themselves.
/// <see cref="Dispose"/> lets an implementation that runs a background timer (e.g. to
/// refresh a rate/ETA display once a second in real time, not only when an image
/// finishes) stop it once the run ends.
/// </summary>
public sealed record LargeCacheProgressReporter(
    Action<string> ReportPhase,
    Action<int, int> ReportOverall,
    Action<int, int> ReportPhaseProgress,
    Action Dispose);

/// <summary>One page's worth of already-joined image data — just what preprocessing needs (id, one blob's bytes, distinct tag names), not the full EF entity graph.</summary>
internal sealed record FetchedPageImage(int ImageId, byte[] BlobData, IReadOnlyList<string> TagNames);

public static class LargeDatasetPreprocessor
{
    private const int DefaultPageSize = 500;
    private const int MaxRetriesPerPage = 5;
    private const int BlobFetchConcurrency = 4;

    public static async Task BuildAsync(
        CoreContext context,
        string outputDirectory,
        int inputSize,
        int? maxImages = null,
        int minImagesPerTag = 15,
        int minTagImageCount = 100,
        LargeCacheProgressReporter? progress = null,
        CancellationToken cancellationToken = default,
        int pageSize = DefaultPageSize,
        int vocabCompactionIntervalPages = 20,
        Func<CoreContext>? contextFactory = null)
    {
        Directory.CreateDirectory(outputDirectory);

        progress?.ReportPhase("Loading tag vocabulary...");
        var vocabularyPath = Path.Combine(outputDirectory, "tag_vocabulary.json");
        var vocabularyDeltaPath = Path.Combine(outputDirectory, "tag_vocabulary.delta.jsonl");

        TagVocabulary vocabulary;
        if (File.Exists(vocabularyPath))
        {
            vocabulary = TagVocabulary.LoadAndCompact(vocabularyPath, vocabularyDeltaPath);
        }
        else
        {
            // Establish an empty base snapshot immediately so every resume (even one
            // that crashes before the run's first compaction) always has a base file to
            // load and replay the delta log onto.
            vocabulary = TagVocabulary.CreateEmpty();
            vocabulary.Save(vocabularyPath);
        }

        // A tag with only a handful of images in the whole corpus isn't worth a
        // trained embedding row yet (CLAUDE.md's minimum-image threshold) — computed
        // as one cheap grouped count over the tag-link table (no blobs touched), and
        // applied uniformly whether or not --max-images caps the run, so a capped
        // sample can't launder an ineligible tag in by never seeing its full count.
        progress?.ReportPhase("Computing per-tag image counts...");
        var eligibleTagNames = await context.ImageImageTags
            .AsNoTracking()
            .GroupBy(link => link.Tag.Name)
            .Where(g => g.Count() >= minTagImageCount)
            .Select(g => g.Key)
            .ToHashSetAsync(cancellationToken);

        IReadOnlySet<int>? selectedIds = null;
        if (maxImages.HasValue)
        {
            // A plain ImageId-ordered Take() has no relationship to tag distribution and
            // can silently drop a tag's images entirely — spend the capped budget on tag
            // coverage instead (see TagCoverageSampleSelector). Explicitly ordered so the
            // selection is reproducible if this run is resumed after a crash — otherwise a
            // resumed run could select a different subset than the original and produce an
            // internally inconsistent cache.
            progress?.ReportPhase("Selecting a tag-coverage sample (scanning the full corpus)...");
            var allImageTags = await context.Images
                .AsNoTracking()
                .OrderBy(i => i.ImageId)
                .Select(i => new { i.ImageId, TagNames = i.TagSources.Select(link => link.Tag.Name).Distinct().ToList() })
                .ToListAsync(cancellationToken);

            selectedIds = TagCoverageSampleSelector.Select(
                allImageTags.Select(i => (i.ImageId, (IReadOnlyList<string>)i.TagNames.Where(eligibleTagNames.Contains).ToList())).ToList(),
                maxImages.Value,
                minImagesPerTag);
        }

        // selectedIds.Count is the exact final image count when capped (the selector
        // never returns more than requested); otherwise a one-off COUNT query gives an
        // accurate total for a real percentage/ETA instead of an open-ended counter.
        if (selectedIds is null)
            progress?.ReportPhase("Counting total images...");
        var total = selectedIds?.Count ?? await context.Images.AsNoTracking().CountAsync(cancellationToken);

        var resumeState = LargeCacheResumeState.Load(outputDirectory);
        progress?.ReportPhase("Opening cache writer...");
        using var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(outputDirectory, inputSize);

        var processed = writer.ImageCount;
        var remaining = maxImages.HasValue ? maxImages.Value - processed : (int?)null;
        var lastImageId = resumeState.LastImageId;
        progress?.ReportOverall(processed, total);

        try
        {
            var pageNumber = 0;
            while (!remaining.HasValue || remaining.Value > 0)
            {
                pageNumber++;
                var take = remaining.HasValue ? Math.Min(pageSize, remaining.Value) : pageSize;

                var page = await FetchPageWithRetry(context, contextFactory, selectedIds, lastImageId, take, pageNumber, progress, cancellationToken);
                if (page.Count == 0)
                    break;

                progress?.ReportPhase($"Page {pageNumber}: decoding {page.Count} images...");

                // Decode/resize/normalize is pure CPU work, independent per image, so it's
                // the one part of the page loop safe to fan out across cores. Vocabulary
                // lookups/mutation and the writer are not thread-safe and stay on the
                // sequential pass below, appending in the page's original order.
                var pixelsByImage = new EncodedImage[page.Count];
                Parallel.For(
                    0,
                    page.Count,
                    new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount, CancellationToken = cancellationToken },
                    i =>
                    {
                        using var blobStream = new MemoryStream(page[i].BlobData);
                        pixelsByImage[i] = ImagePreprocessing.LoadAndEncode(blobStream, inputSize);
                    });

                progress?.ReportPhase($"Page {pageNumber}: writing {page.Count} images to cache...");
                for (var i = 0; i < page.Count; i++)
                {
                    var image = page[i];
                    var tagRows = new List<int>();
                    foreach (var tagName in image.TagNames)
                    {
                        if (!eligibleTagNames.Contains(tagName))
                            continue;

                        var record = vocabulary.RecordObservation(tagName);
                        tagRows.Add(record.RowIndex);
                    }

                    writer.Append(pixelsByImage[i], tagRows);

                    processed++;
                    progress?.ReportOverall(processed, total);
                    progress?.ReportPhaseProgress(i + 1, page.Count);
                }

                lastImageId = page[^1].ImageId;
                remaining -= page.Count;

                // Persist everything after each page so a crash (e.g. a dropped DB
                // connection over a long WAN-backed run) loses at most one page's worth
                // of work, and re-running the same command resumes right here. New tag
                // rows and image-count updates are appended to a delta log rather than
                // rewriting the whole vocabulary file: that rewrite's cost scales with
                // vocabulary size, and redoing it every ~500 images is what made a long
                // build visibly slow down over time as the vocabulary grew toward
                // CLAUDE.md's hundreds-of-thousands-of-tags scale. The vocabulary delta is saved
                // before the cache is flushed: the cache's tag rows reference RowIndex
                // values that only exist once the delta durably records them, so if a
                // crash lands between these two calls, the ordering guarantees the cache
                // never ends up referencing a row the vocabulary doesn't know about yet.
                progress?.ReportPhase($"Page {pageNumber}: checkpointing (cache, vocabulary, resume state)...");
                vocabulary.SaveDelta(vocabularyDeltaPath);
                writer.Flush();
                new LargeCacheResumeState(lastImageId).Save(outputDirectory);

                // Periodically fold the delta log back into a fresh tag_vocabulary.json
                // (rather than only at the very end) so a tool reading the vocabulary
                // file directly — or a run that's killed rather than crashing outright —
                // sees a reasonably fresh snapshot without paying the full-rewrite cost
                // on every single page.
                if (pageNumber % vocabCompactionIntervalPages == 0)
                {
                    progress?.ReportPhase($"Page {pageNumber}: compacting vocabulary...");
                    vocabulary.Save(vocabularyPath);
                    File.Delete(vocabularyDeltaPath);
                }

                if (page.Count < take)
                    break;
            }

            // Compact the delta into a fresh, fully up-to-date snapshot now that the run
            // has finished, so the next Load starts clean.
            progress?.ReportPhase("Compacting vocabulary...");
            vocabulary.Save(vocabularyPath);
            File.Delete(vocabularyDeltaPath);

            progress?.ReportPhase("Done.");
        }
        finally
        {
            // Stops a real-time reporter's background refresh timer (if any) — without
            // this it would keep firing (and touching a progress row that no longer
            // exists) after this method returns, on both the success and error paths.
            progress?.Dispose();
        }
    }

    private static async Task<List<FetchedPageImage>> FetchPageWithRetry(
        CoreContext context,
        Func<CoreContext>? contextFactory,
        IReadOnlySet<int>? selectedIds,
        int? lastImageId,
        int pageSize,
        int pageNumber,
        LargeCacheProgressReporter? progress,
        CancellationToken cancellationToken)
    {
        for (var attempt = 1; ; attempt++)
        {
            try
            {
                return await FetchPage(context, contextFactory, selectedIds, lastImageId, pageSize, pageNumber, attempt, progress, cancellationToken);
            }
            catch (SqlException ex) when (attempt < MaxRetriesPerPage)
            {
                var delay = TimeSpan.FromSeconds(Math.Pow(2, attempt));
                var message = $"Page {pageNumber}: transient database error (attempt {attempt}/{MaxRetriesPerPage}): {ex.Message} — retrying in {delay.TotalSeconds:F0}s.";
                if (progress is not null)
                    progress.ReportPhase(message);
                else
                    Console.Error.WriteLine(message);
                await Task.Delay(delay, cancellationToken);
            }
        }
    }

    /// <summary>
    /// Fetches one page as three targeted queries (image ids, then blobs, then tags)
    /// instead of one query with two Included collections under <c>AsSplitQuery</c>. EF
    /// already issues the same three round trips under the hood for that shape — doing
    /// them explicitly here lets each one report its own phase, and — the actual point
    /// of the split — lets the blob fetch (raw image bytes, the one that dominates a
    /// page's wall time by a wide margin) run in parallel across several connections,
    /// and the independent tag fetch run concurrently with it, instead of three
    /// round trips serialized one after another on a single connection.
    /// </summary>
    private static async Task<List<FetchedPageImage>> FetchPage(
        CoreContext context,
        Func<CoreContext>? contextFactory,
        IReadOnlySet<int>? selectedIds,
        int? lastImageId,
        int pageSize,
        int pageNumber,
        int attempt,
        LargeCacheProgressReporter? progress,
        CancellationToken cancellationToken)
    {
        var attemptSuffix = attempt == 1 ? "" : $" (attempt {attempt}/{MaxRetriesPerPage})";

        progress?.ReportPhase($"Page {pageNumber}: fetching image list...{attemptSuffix}");
        IQueryable<Image> idQuery = context.Images.AsNoTracking().OrderBy(i => i.ImageId);
        if (selectedIds is not null)
            idQuery = idQuery.Where(i => selectedIds.Contains(i.ImageId));
        if (lastImageId.HasValue)
            idQuery = idQuery.Where(i => i.ImageId > lastImageId.Value);

        var imageIds = await idQuery.Select(i => i.ImageId).Take(pageSize).ToListAsync(cancellationToken);
        if (imageIds.Count == 0)
            return [];

        Dictionary<int, byte[]> blobByImageId;
        Dictionary<int, IReadOnlyList<string>> tagNamesByImageId;
        if (contextFactory is not null)
        {
            // context is free to use for tags concurrently with the blob fetch only
            // because the blob fetch itself never touches context here — it runs
            // entirely on fresh connections from contextFactory. A DbContext can't have
            // two operations in flight at once, so this ordering isn't optional.
            progress?.ReportPhase($"Page {pageNumber}: fetching {imageIds.Count} image blobs (parallel) and tags...{attemptSuffix}");
            var tagsTask = FetchTags(context, imageIds, cancellationToken);
            blobByImageId = await FetchBlobsParallel(contextFactory, imageIds, progress, cancellationToken);
            tagNamesByImageId = await tagsTask;
        }
        else
        {
            progress?.ReportPhase($"Page {pageNumber}: fetching {imageIds.Count} image blobs...{attemptSuffix}");
            blobByImageId = await FetchBlobs(context, imageIds, cancellationToken);

            progress?.ReportPhase($"Page {pageNumber}: fetching tags...{attemptSuffix}");
            tagNamesByImageId = await FetchTags(context, imageIds, cancellationToken);
        }

        return imageIds
            .Where(blobByImageId.ContainsKey)
            .Select(id => new FetchedPageImage(id, blobByImageId[id], tagNamesByImageId.GetValueOrDefault(id, [])))
            .ToList();
    }

    /// <summary>
    /// Splits <paramref name="imageIds"/> into up to <see cref="BlobFetchConcurrency"/>
    /// chunks and fetches each chunk's blobs on its own connection concurrently. A
    /// single connection often can't saturate available network throughput for bulk
    /// binary transfer on its own (per-connection/per-query overhead, not just raw
    /// bandwidth, tends to cap one connection well below the link's real capacity) —
    /// splitting the same total transfer across several connections is a standard way
    /// to claw back that gap. Connections come from <paramref name="contextFactory"/>
    /// rather than sharing <c>context</c>, since EF Core forbids concurrent operations
    /// on one DbContext.
    ///
    /// Each chunk finishing is also the one genuinely countable step inside "fetching" —
    /// the id-list and tag queries are each a single all-or-nothing round trip — so this
    /// reports 0/N, 1/N, ... N/N as chunks land, giving the phase row a real percentage
    /// during the sub-step that otherwise dominates a page's wall time.
    /// </summary>
    private static async Task<Dictionary<int, byte[]>> FetchBlobsParallel(
        Func<CoreContext> contextFactory,
        IReadOnlyList<int> imageIds,
        LargeCacheProgressReporter? progress,
        CancellationToken cancellationToken)
    {
        var chunkSize = Math.Max(1, (int)Math.Ceiling(imageIds.Count / (double)BlobFetchConcurrency));
        var chunks = imageIds.Chunk(chunkSize).ToList();
        var completedChunks = 0;
        progress?.ReportPhaseProgress(0, chunks.Count);

        var chunkResults = await Task.WhenAll(chunks.Select(async chunk =>
        {
            await using var chunkContext = contextFactory();
            var result = await FetchBlobs(chunkContext, chunk, cancellationToken);
            progress?.ReportPhaseProgress(Interlocked.Increment(ref completedChunks), chunks.Count);
            return result;
        }));

        var merged = new Dictionary<int, byte[]>();
        foreach (var chunkResult in chunkResults)
            foreach (var (imageId, data) in chunkResult)
                merged[imageId] = data;
        return merged;
    }

    private static async Task<Dictionary<int, byte[]>> FetchBlobs(CoreContext context, IReadOnlyList<int> imageIds, CancellationToken cancellationToken)
    {
        var blobs = await context.ImageBlobs
            .AsNoTracking()
            .Where(b => imageIds.Contains(b.Image.ImageId))
            .Select(b => new { ImageId = b.Image.ImageId, b.Data })
            .ToListAsync(cancellationToken);
        return blobs.GroupBy(b => b.ImageId).ToDictionary(g => g.Key, g => g.First().Data);
    }

    private static async Task<Dictionary<int, IReadOnlyList<string>>> FetchTags(CoreContext context, IReadOnlyList<int> imageIds, CancellationToken cancellationToken)
    {
        var tagLinks = await context.ImageImageTags
            .AsNoTracking()
            .Where(link => imageIds.Contains(link.ImagesImageId))
            .Select(link => new { link.ImagesImageId, TagName = link.Tag.Name })
            .ToListAsync(cancellationToken);
        return tagLinks
            .GroupBy(t => t.ImagesImageId)
            .ToDictionary(g => g.Key, g => (IReadOnlyList<string>)g.Select(t => t.TagName).Distinct().ToList());
    }
}
