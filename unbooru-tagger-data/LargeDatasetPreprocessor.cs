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
/// rather than a capped sample.
/// </summary>
public static class LargeDatasetPreprocessor
{
    public static async Task BuildAsync(
        CoreContext context,
        string outputDirectory,
        int inputSize,
        int? maxImages = null,
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
            query = query.Take(maxImages.Value);

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
