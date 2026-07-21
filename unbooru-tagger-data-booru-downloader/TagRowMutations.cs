using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Crawler;

/// <summary>
/// The two ways an image's live tag-row-index set (<c>CrawlWorkingState.ImageTagRowsByCacheRow</c>
/// in a normal crawl, or the equivalent working set in <see cref="TagRefresher"/>) ever
/// changes after the row is first written, kept in one place so both call sites credit/
/// debit <see cref="TagVocabulary"/>'s per-tag <c>ImageCount</c> exactly once per
/// (image, tag) association — the bug this was extracted to fix: crediting on every
/// re-observation of an already-present tag silently inflates <c>ImageCount</c> every
/// time the same image is re-listed by a later crawl or refresh pass.
/// </summary>
internal static class TagRowMutations
{
    /// <summary>
    /// Adds <paramref name="tagName"/> to <paramref name="currentTagRows"/> if it isn't
    /// already there, creating its vocabulary row on first-ever observation. Only
    /// credits <see cref="TagVocabulary"/>'s <c>ImageCount</c> when this is genuinely
    /// new to this image — an already-present tag is a no-op. Returns whether it was
    /// added.
    /// </summary>
    public static bool TryAddTagToImage(TagVocabulary vocabulary, string tagName, HashSet<int> currentTagRows)
    {
        if (vocabulary.TryGet(tagName, out var existingRecord) && currentTagRows.Contains(existingRecord.RowIndex))
            return false;

        var record = vocabulary.RecordObservation(tagName);
        return currentTagRows.Add(record.RowIndex);
    }

    /// <summary>Removes a tag row from <paramref name="currentTagRows"/> if present, debiting <see cref="TagVocabulary"/>'s <c>ImageCount</c> to match. A no-op if the tag wasn't on this image.</summary>
    public static void RemoveTagFromImage(TagVocabulary vocabulary, int rowIndex, HashSet<int> currentTagRows)
    {
        if (currentTagRows.Remove(rowIndex))
            vocabulary.AdjustImageCount(vocabulary.GetByRowIndex(rowIndex), -1);
    }
}
