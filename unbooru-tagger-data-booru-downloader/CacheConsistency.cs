namespace UnbooruTagger.Crawler;

/// <summary>
/// Verifies <c>crawl.sqlite</c>'s <c>Images</c> table actually agrees with the cache
/// files (<c>images.bin</c>/<c>tag_rows.jsonl</c>) it's supposed to index, before
/// <see cref="DatasetCrawler"/> or <see cref="TagRefresher"/> trusts
/// <c>CacheRowIndex</c> as a reliable key into either. The two are meant to stay in
/// lockstep purely by convention (crawl.sqlite is only ever written with row indices
/// the cache writer itself just handed out) — nothing in either file format enforces
/// it. If the cache files are ever deleted, moved, or replaced while crawl.sqlite is
/// kept (or vice versa), a later run's cache writer restarts counting from a lower
/// <c>ImageCount</c> than crawl.sqlite believes exists, and silently reassigns an
/// already-claimed row index to an unrelated new image — <c>Images.Md5</c> is the only
/// uniqueness constraint the schema enforces, so nothing stops two different images
/// from both claiming the same <c>CacheRowIndex</c>. That's silent, active corruption
/// (one image's tag_rows.jsonl row now holds a different image's labels), worse than
/// the "wasted work, not corruption" a mid-checkpoint crash costs — so this throws
/// instead of letting either caller build a dedup/tag-row index on top of it.
/// </summary>
public static class CacheConsistency
{
    public static void Validate(
        IReadOnlyList<(string Md5, int CacheRowIndex, ulong PHash)> existingImages,
        int writerImageCount,
        string outputDirectory)
    {
        var seenBy = new Dictionary<int, string>();
        foreach (var image in existingImages)
        {
            if (image.CacheRowIndex < 0 || image.CacheRowIndex >= writerImageCount)
                throw new InvalidDataException(
                    $"crawl.sqlite references cache row {image.CacheRowIndex} for image '{image.Md5}', but the cache " +
                    $"files in '{outputDirectory}' only have {writerImageCount} row(s). The cache files and " +
                    "crawl.sqlite have fallen out of sync — most likely images.bin/tag_rows.jsonl were deleted, " +
                    "moved, or replaced while crawl.sqlite was kept. Refusing to continue: doing so would silently " +
                    "reassign that row to a different image and corrupt its tag labels. Restore the original " +
                    "images.bin/tag_rows.jsonl next to this crawl.sqlite, or start a fresh --output-dir if the cache " +
                    "files are genuinely gone.");

            if (seenBy.TryGetValue(image.CacheRowIndex, out var otherMd5))
                throw new InvalidDataException(
                    $"crawl.sqlite has two different images ('{otherMd5}' and '{image.Md5}') both claiming cache row " +
                    $"{image.CacheRowIndex} in '{outputDirectory}'. The cache files and crawl.sqlite have fallen out " +
                    "of sync — most likely images.bin/tag_rows.jsonl were deleted, moved, or replaced while " +
                    "crawl.sqlite was kept, so a later run reassigned an already-claimed row to a new image. " +
                    "Refusing to continue: one of these images' labels in tag_rows.jsonl is now wrong and can't be " +
                    "disambiguated automatically.");

            seenBy[image.CacheRowIndex] = image.Md5;
        }
    }
}
