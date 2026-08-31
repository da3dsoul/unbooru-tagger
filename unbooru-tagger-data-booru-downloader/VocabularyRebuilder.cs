using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Crawler;

/// <summary>Summary counts for a <see cref="VocabularyRebuilder.Rebuild"/> pass, printed by <c>rebuild-vocabulary</c> before/instead of writing anything.</summary>
public sealed record VocabularyRebuildResult(
    int SeedTagCount,
    int FinalTagCount,
    int ConfirmedImageCount,
    int ImagesWithSourceData,
    int ImagesWithoutSourceData);

/// <summary>
/// Disaster recovery for a lost/corrupted <c>tag_vocabulary.json</c>: re-derives the
/// tag-string &lt;-&gt; row-index mapping (and each image's tag-row indices) from
/// <c>crawl.sqlite</c>'s own <c>ImageSources.Tags</c> snapshots — the same ground truth
/// <c>refresh-tags</c> reconciles against — rather than trusting anything already in
/// <c>tag_rows.sqlite</c>, which is unreadable without the vocabulary that produced it.
///
/// <paramref name="seedVocabulary"/> (see <c>rebuild-vocabulary</c>'s
/// <c>--seed-vocabulary</c>) should be the most recent known-good vocabulary snapshot
/// available — e.g. a training checkpoint's own <c>tag_vocabulary.json</c>, which
/// <c>Checkpoint.Save</c> writes as a byte-for-byte copy of whatever the dataset's
/// vocabulary was at that moment. Every tag it already contains keeps its exact
/// <see cref="TagRecord.RowIndex"/>: <see cref="TagVocabulary.RecordObservation"/> only
/// ever appends past the current row count, never reorders or reuses one, so a past
/// snapshot is always a verified-correct prefix of any later vocabulary built from the
/// same crawl. That matters beyond just correctness here — if a training checkpoint's
/// <c>tag_embeddings.bin</c> was trained against those row indices, this is what keeps
/// them valid after the rebuild. Only tags first observed after the seed snapshot was
/// taken get freshly appended rows, in whatever order this pass happens to encounter
/// them in -- fine, since nothing has trained against those rows yet.
///
/// Every <see cref="TagRecord.ImageCount"/> is recomputed from scratch (reset to zero,
/// then re-credited once per image actually carrying that tag here) rather than kept
/// from the seed snapshot, so counts end up correct regardless of how much the corpus
/// grew between the seed snapshot and now -- no need to know exactly which image the
/// seed snapshot was taken at.
///
/// Note that <c>ImageSources.Tags</c> holds already-resolved tag IDENTITIES (it's written
/// from <see cref="DatasetCrawler"/>'s own <c>eligibleTagsOnPost</c>), not the raw booru
/// names a live post carries — so a non-General tag reads as <c>character:frieren</c>
/// there, not <c>frieren</c>. <see cref="TagRowMutations.EligibleIdentities"/> normalizes
/// either shape back to a raw name before resolving it against the current survey; see its
/// doc comment for why that's both necessary here and a no-op for the live-crawl callers.
///
/// An image with no <c>ImageSources</c> row at all (crawl.sqlite itself is missing that
/// bookkeeping -- see <c>rebuild-vocabulary</c>'s own remarks on why a handful of
/// recently-committed images can end up in this state after a crash) gets an empty tag
/// row: there is nothing left anywhere to recover its tags from, and leaving whatever
/// stale integers happen to already be in <c>tag_rows.sqlite</c> would silently point at
/// the wrong tags once renumbered, which is worse than no label at all.
/// </summary>
public static class VocabularyRebuilder
{
    public static async Task<(IReadOnlyList<(int CacheRowIndex, IReadOnlyList<int> TagRows)> Rows, VocabularyRebuildResult Result)> Rebuild(
        CrawlDatabase db,
        TagVocabulary seedVocabulary,
        int minImages,
        TagExclusionRules? excludedTags,
        IReadOnlyDictionary<string, string>? tagAliases,
        int confirmedImageCount,
        Action<string>? onPhase = null)
    {
        var seedTagCount = seedVocabulary.Records.Count;
        foreach (var record in seedVocabulary.Records)
            record.ImageCount = 0;

        onPhase?.Invoke("Loading tag survey...");
        var allTags = await db.GetAllSurveyedTagsAsync().ConfigureAwait(false);
        var eligibleTags = allTags.Where(t => TagEligibility.IsEligible(t, minImages, excludedTags));
        var eligibleTagIdentities = TagRowMutations.BuildEligibleIdentities(eligibleTags, tagAliases);

        onPhase?.Invoke("Loading every image's known-source tag snapshots...");
        var eligibleTagsByMd5 = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        foreach (var (md5, tags) in await db.GetAllImageSourceTagsAsync().ConfigureAwait(false))
        {
            if (tags is null)
                continue; // never captured -- not the same as a confirmed-empty tag list

            if (!eligibleTagsByMd5.TryGetValue(md5, out var set))
                eligibleTagsByMd5[md5] = set = new HashSet<string>(StringComparer.Ordinal);

            foreach (var identity in TagRowMutations.EligibleIdentities(tags, eligibleTagIdentities))
                set.Add(identity);
        }

        onPhase?.Invoke("Loading image list...");
        var images = await db.GetAllImagesAsync().ConfigureAwait(false);

        onPhase?.Invoke("Replaying every image's tags in cache-row order...");
        var rows = new (int CacheRowIndex, IReadOnlyList<int> TagRows)[confirmedImageCount];
        var covered = new bool[confirmedImageCount];
        var imagesWithSourceData = 0;

        foreach (var (md5, cacheRowIndex, _) in images.OrderBy(i => i.CacheRowIndex))
        {
            if (cacheRowIndex < 0 || cacheRowIndex >= confirmedImageCount || covered[cacheRowIndex])
                continue; // out-of-range or duplicate bookkeeping -- shouldn't happen, but never trust blindly

            covered[cacheRowIndex] = true;

            IReadOnlyList<int> tagRowIndices = [];
            if (eligibleTagsByMd5.TryGetValue(md5, out var eligibleTagNames))
            {
                imagesWithSourceData++;
                tagRowIndices = eligibleTagNames.Select(tag => seedVocabulary.RecordObservation(tag).RowIndex).ToList();
            }

            rows[cacheRowIndex] = (cacheRowIndex, tagRowIndices);
        }

        for (var i = 0; i < confirmedImageCount; i++)
            if (!covered[i])
                rows[i] = (i, []); // pixel-file/tag_rows.sqlite has this row, crawl.sqlite doesn't -- nothing to recover

        var result = new VocabularyRebuildResult(
            SeedTagCount: seedTagCount,
            FinalTagCount: seedVocabulary.Records.Count,
            ConfirmedImageCount: confirmedImageCount,
            ImagesWithSourceData: imagesWithSourceData,
            ImagesWithoutSourceData: confirmedImageCount - imagesWithSourceData);

        return (rows, result);
    }
}
