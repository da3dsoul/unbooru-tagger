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
/// </summary>
public static class LargeDatasetPreprocessor
{
    public static async Task BuildAsync(
        CoreContext context,
        string outputDirectory,
        int inputSize,
        int? maxImages = null,
        int minImagesPerTag = 15,
        IProgress<int>? progress = null,
        CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(outputDirectory);
        var vocabulary = TagVocabulary.CreateEmpty();

        IQueryable<Image> query = context.Images
            .Include(i => i.Blobs)
            .Include(i => i.TagSources).ThenInclude(link => link.Tag)
            .AsSplitQuery()
            .AsNoTracking()
            .OrderBy(i => i.ImageId);

        if (maxImages.HasValue)
        {
            // A plain ImageId-ordered Take() has no relationship to tag distribution and
            // can silently drop a tag's images entirely — spend the capped budget on tag
            // coverage instead (see TagCoverageSampleSelector).
            var allImageTags = await context.Images
                .AsNoTracking()
                .Select(i => new { i.ImageId, TagNames = i.TagSources.Select(link => link.Tag.Name).Distinct().ToList() })
                .ToListAsync(cancellationToken);

            var selectedIds = TagCoverageSampleSelector.Select(
                allImageTags.Select(i => (i.ImageId, (IReadOnlyList<string>)i.TagNames)).ToList(),
                maxImages.Value,
                minImagesPerTag);

            query = query.Where(i => selectedIds.Contains(i.ImageId));
        }

        using var writer = new PreprocessedDatasetCacheWriter(outputDirectory, inputSize);

        var processed = 0;
        await foreach (var image in query.AsAsyncEnumerable().WithCancellation(cancellationToken))
        {
            var tagRows = new List<int>();
            foreach (var tagName in image.TagSources.Select(link => link.Tag.Name).Distinct())
            {
                if (!vocabulary.TryGet(tagName, out var record))
                    record = vocabulary.AddTag(tagName);
                record.ImageCount++;
                tagRows.Add(record.RowIndex);
            }

            using var blobStream = new MemoryStream(image.Blobs[0].Data);
            var pixels = ImagePreprocessing.LoadAndNormalize(blobStream, inputSize);
            writer.Append(pixels, tagRows);

            processed++;
            progress?.Report(processed);
        }

        vocabulary.Save(Path.Combine(outputDirectory, "tag_vocabulary.json"));
    }
}
