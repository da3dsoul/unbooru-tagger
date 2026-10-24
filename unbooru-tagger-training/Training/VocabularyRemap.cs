using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Maps a dataset's tag rows onto the vocabulary a model actually uses, by tag name.
/// Matching by name rather than by row position is what lets the model's vocabulary differ
/// from the dataset's: a fresh run drops rows no image carries (<see cref="Compact"/> — the
/// crawled vocabulary has thousands of stale zero-image rows, e.g. "long hair" beside the
/// live "long_hair"), and a checkpoint can carry rows the dataset doesn't (tags added later
/// with add-tag). Tags the target vocabulary lacks are dropped from the image's rows.
/// </summary>
public static class VocabularyRemap
{
    /// <summary>A vocabulary of only the tags at least one image carries, in their original order, with ImageCount set from <paramref name="imageTagRows"/>.</summary>
    public static TagVocabulary Compact(TagVocabulary vocabulary, IReadOnlyList<IReadOnlyList<int>> imageTagRows)
    {
        var counts = new int[vocabulary.Records.Count];
        foreach (var rows in imageTagRows)
            foreach (var row in rows)
                if ((uint)row < (uint)counts.Length)
                    counts[row]++;

        var compact = TagVocabulary.CreateEmpty();
        for (var row = 0; row < counts.Length; row++)
            if (counts[row] > 0)
            {
                var record = compact.AddTag(vocabulary.GetByRowIndex(row).Tag);
                record.ImageCount = counts[row];
            }

        return compact;
    }

    /// <returns>Each image's rows re-expressed in <paramref name="target"/>'s row numbering, and how many distinct source tags had no match.</returns>
    public static (List<List<int>> Rows, int UnmatchedTags) Apply(TagVocabulary source, IReadOnlyList<IReadOnlyList<int>> imageTagRows, TagVocabulary target)
    {
        var map = new int[source.Records.Count];
        var unmatched = 0;
        for (var row = 0; row < map.Length; row++)
        {
            map[row] = target.TryGet(source.GetByRowIndex(row).Tag, out var record) ? record.RowIndex : -1;
            if (map[row] < 0)
                unmatched++;
        }

        var result = new List<List<int>>(imageTagRows.Count);
        foreach (var rows in imageTagRows)
        {
            var mapped = new List<int>(rows.Count);
            foreach (var row in rows)
                if ((uint)row < (uint)map.Length && map[row] >= 0)
                    mapped.Add(map[row]);
            result.Add(mapped);
        }

        return (result, unmatched);
    }
}
