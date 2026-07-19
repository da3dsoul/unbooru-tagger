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
/// right where it left off instead of restarting from scratch. New tag rows are
/// checkpointed via <see cref="TagVocabulary.SaveDelta"/> (append-only) rather than
/// re-serializing the whole vocabulary every page, so a run's per-page cost doesn't
/// grow with vocabulary size as it scales toward CLAUDE.md's long-tail vocabulary.
///
/// Within a page, decode/resize/normalize (CPU-bound, independent per image) runs in
/// parallel across cores; vocabulary bookkeeping and cache writes are cheap by
/// comparison and stay sequential to avoid needing to synchronize either one.
/// </summary>
public static class LargeDatasetPreprocessor
{
    private const int DefaultPageSize = 500;
    private const int MaxRetriesPerPage = 5;

    public static async Task BuildAsync(
        CoreContext context,
        string outputDirectory,
        int inputSize,
        int? maxImages = null,
        int minImagesPerTag = 15,
        IProgress<int>? progress = null,
        CancellationToken cancellationToken = default,
        int pageSize = DefaultPageSize)
    {
        Directory.CreateDirectory(outputDirectory);

        var vocabularyPath = Path.Combine(outputDirectory, "tag_vocabulary.json");
        var vocabularyDeltaPath = Path.Combine(outputDirectory, "tag_vocabulary.delta.jsonl");

        TagVocabulary vocabulary;
        if (File.Exists(vocabularyPath))
        {
            vocabulary = TagVocabulary.Load(vocabularyPath, vocabularyDeltaPath);
        }
        else
        {
            // Establish an empty base snapshot immediately so every resume (even one
            // that crashes before the run's first compaction) always has a base file to
            // load and replay the delta log onto.
            vocabulary = TagVocabulary.CreateEmpty();
            vocabulary.Save(vocabularyPath);
        }

        IReadOnlySet<int>? selectedIds = null;
        if (maxImages.HasValue)
        {
            // A plain ImageId-ordered Take() has no relationship to tag distribution and
            // can silently drop a tag's images entirely — spend the capped budget on tag
            // coverage instead (see TagCoverageSampleSelector). Explicitly ordered so the
            // selection is reproducible if this run is resumed after a crash — otherwise a
            // resumed run could select a different subset than the original and produce an
            // internally inconsistent cache.
            var allImageTags = await context.Images
                .AsNoTracking()
                .OrderBy(i => i.ImageId)
                .Select(i => new { i.ImageId, TagNames = i.TagSources.Select(link => link.Tag.Name).Distinct().ToList() })
                .ToListAsync(cancellationToken);

            selectedIds = TagCoverageSampleSelector.Select(
                allImageTags.Select(i => (i.ImageId, (IReadOnlyList<string>)i.TagNames)).ToList(),
                maxImages.Value,
                minImagesPerTag);
        }

        var resumeState = LargeCacheResumeState.Load(outputDirectory);
        using var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(outputDirectory, inputSize);

        var processed = writer.ImageCount;
        var remaining = maxImages.HasValue ? maxImages.Value - processed : (int?)null;
        var lastImageId = resumeState.LastImageId;

        while (!remaining.HasValue || remaining.Value > 0)
        {
            var take = remaining.HasValue ? Math.Min(pageSize, remaining.Value) : pageSize;
            var page = await FetchPageWithRetry(context, selectedIds, lastImageId, take, cancellationToken);
            if (page.Count == 0)
                break;

            // Decode/resize/normalize is pure CPU work, independent per image, so it's
            // the one part of the page loop safe to fan out across cores. Vocabulary
            // lookups/mutation and the writer are not thread-safe and stay on the
            // sequential pass below, appending in the page's original order.
            var pixelsByImage = new float[page.Count][];
            Parallel.For(
                0,
                page.Count,
                new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount, CancellationToken = cancellationToken },
                i =>
                {
                    using var blobStream = new MemoryStream(page[i].Blobs[0].Data);
                    pixelsByImage[i] = ImagePreprocessing.LoadAndNormalize(blobStream, inputSize);
                });

            for (var i = 0; i < page.Count; i++)
            {
                var image = page[i];
                var tagRows = new List<int>();
                foreach (var tagName in image.TagSources.Select(link => link.Tag.Name).Distinct())
                {
                    if (!vocabulary.TryGet(tagName, out var record))
                        record = vocabulary.AddTag(tagName);
                    record.ImageCount++;
                    tagRows.Add(record.RowIndex);
                }

                writer.Append(pixelsByImage[i], tagRows);

                processed++;
                progress?.Report(processed);
            }

            lastImageId = page[^1].ImageId;
            remaining -= page.Count;

            // Persist everything after each page so a crash (e.g. a dropped DB
            // connection over a long WAN-backed run) loses at most one page's worth
            // of work, and re-running the same command resumes right here. New tag
            // rows are appended to a delta log rather than rewriting the whole
            // vocabulary file: that rewrite's cost scales with vocabulary size, and
            // redoing it every ~500 images is what made a long build visibly slow
            // down over time as the vocabulary grew toward CLAUDE.md's
            // hundreds-of-thousands-of-tags scale.
            writer.Flush();
            vocabulary.SaveDelta(vocabularyDeltaPath);
            new LargeCacheResumeState(lastImageId).Save(outputDirectory);

            if (page.Count < take)
                break;
        }

        // Compact the delta into a fresh, fully up-to-date snapshot now that the run
        // has finished, so the next Load starts clean (also picking up ImageCount
        // updates for tags that already existed, which SaveDelta doesn't persist).
        vocabulary.Save(vocabularyPath);
        File.Delete(vocabularyDeltaPath);
    }

    private static async Task<List<Image>> FetchPageWithRetry(
        CoreContext context,
        IReadOnlySet<int>? selectedIds,
        int? lastImageId,
        int pageSize,
        CancellationToken cancellationToken)
    {
        for (var attempt = 1; ; attempt++)
        {
            try
            {
                IQueryable<Image> query = context.Images
                    .Include(i => i.Blobs)
                    .Include(i => i.TagSources).ThenInclude(link => link.Tag)
                    .AsSplitQuery()
                    .AsNoTracking()
                    .OrderBy(i => i.ImageId);

                if (selectedIds is not null)
                    query = query.Where(i => selectedIds.Contains(i.ImageId));
                if (lastImageId.HasValue)
                    query = query.Where(i => i.ImageId > lastImageId.Value);

                return await query.Take(pageSize).ToListAsync(cancellationToken);
            }
            catch (SqlException ex) when (attempt < MaxRetriesPerPage)
            {
                var delay = TimeSpan.FromSeconds(Math.Pow(2, attempt));
                Console.Error.WriteLine($"Transient database error fetching page after ImageId {lastImageId} (attempt {attempt}/{MaxRetriesPerPage}): {ex.Message} — retrying in {delay.TotalSeconds:F0}s.");
                await Task.Delay(delay, cancellationToken);
            }
        }
    }
}
