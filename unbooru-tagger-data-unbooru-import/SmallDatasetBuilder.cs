using Microsoft.EntityFrameworkCore;
using unbooru.Core;
using UnbooruTagger.Core.Dataset;

namespace UnbooruTagger.Data;

/// <summary>
/// Small-mode dataset builder: pulls images tagged with any of the target tags, plus
/// an equal-sized sample of images without a full match. When more images match than
/// fit under a max-images cap, the leftover matches (which still share 1-to-all of the
/// target tags) are mixed into that negative sample alongside true zero-overlap images,
/// so a quick add-tag fine-tune sees a spread of "not quite this" examples rather than
/// only pure background — falling back to zero-overlap-only when there's no leftover.
/// </summary>
public static class SmallDatasetBuilder
{
    public static async Task<DatasetManifest> BuildAsync(
        CoreContext context,
        IReadOnlyList<string> targetTags,
        string outputDirectory,
        int? maxImages = null,
        Random? random = null,
        IProgress<ImageBuildProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var targetTagIds = await context.ImageTags
            .Where(t => targetTags.Contains(t.Name))
            .Select(t => t.ImageTagId)
            .ToListAsync(cancellationToken);

        if (targetTagIds.Count == 0)
            throw new InvalidOperationException($"None of the target tags ({string.Join(", ", targetTags)}) were found in unbooru's tag table.");

        var matchingIds = await context.ImageImageTags
            .Where(link => targetTagIds.Contains(link.TagsImageTagId))
            .Select(link => link.ImagesImageId)
            .Distinct()
            .ToListAsync(cancellationToken);

        var zeroOverlapIds = await context.Images
            .Where(i => !i.TagSources.Any(link => targetTagIds.Contains(link.TagsImageTagId)))
            .Select(i => i.ImageId)
            .ToListAsync(cancellationToken);

        var (selectedPositiveIds, selectedNegativeIds) = BalancedSampleSelector.Select(matchingIds, zeroOverlapIds, maxImages, random);
        var selectedIds = selectedPositiveIds.Concat(selectedNegativeIds).ToList();

        var images = await context.Images
            .Where(i => selectedIds.Contains(i.ImageId))
            .Include(i => i.Blobs)
            .Include(i => i.TagSources).ThenInclude(link => link.Tag)
            .AsNoTracking()
            .ToListAsync(cancellationToken);

        var imagesDirectory = Path.Combine(outputDirectory, "images");
        Directory.CreateDirectory(imagesDirectory);

        var entries = new List<DatasetImageEntry>();
        var written = 0;
        foreach (var image in images)
        {
            // Extension doesn't matter — SkiaSharp sniffs format from content, and
            // unbooru's ImageBlob has no stored format/extension column anyway.
            var imagePath = Path.Combine(imagesDirectory, image.ImageId.ToString());
            await File.WriteAllBytesAsync(imagePath, image.Blobs[0].Data, cancellationToken);

            var tags = image.TagSources.Select(link => link.Tag.Name).Distinct().ToList();
            entries.Add(new DatasetImageEntry(imagePath, tags));

            written++;
            progress?.Report(new ImageBuildProgress(written, images.Count));
        }

        var manifest = new DatasetManifest(entries);
        manifest.Save(Path.Combine(outputDirectory, "manifest.json"));
        return manifest;
    }
}
