using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Folds a dataset's <see cref="ImageRatings"/> into training as the three cumulative labels
/// (<see cref="ImageRatings.CumulativeTags"/>): appended to the vocabulary if missing, and
/// added to each image's tag rows up to its rating's level. Ratings never lived in the
/// crawled tag rows (the old <c>rating:*</c> vocabulary rows have no images), so before this
/// the model had no way to learn them at all.
/// </summary>
public static class RatingLabels
{
    /// <returns>The vocabulary rows of the three cumulative labels, lowest level first.</returns>
    public static int[] EnsureRows(TagVocabulary vocabulary) =>
        ImageRatings.CumulativeTags
            .Select(tag => vocabulary.TryGet(tag, out var record) ? record.RowIndex : vocabulary.AddTag(tag).RowIndex)
            .ToArray();

    /// <summary>Null if the vocabulary has none of the cumulative rows (a model trained before ratings existed).</summary>
    public static int[]? FindRows(TagVocabulary vocabulary)
    {
        var rows = ImageRatings.CumulativeTags.Select(tag => vocabulary.TryGet(tag, out var record) ? record.RowIndex : -1).ToArray();
        return rows.All(r => r >= 0) ? rows : null;
    }

    /// <summary>Copies of <paramref name="imageTagRows"/> with each image's cumulative rating rows appended.</summary>
    public static List<List<int>> Apply(IReadOnlyList<IReadOnlyList<int>> imageTagRows, IReadOnlyList<ContentRating> ratings, int[] cumulativeRows)
    {
        var result = new List<List<int>>(imageTagRows.Count);
        for (var i = 0; i < imageTagRows.Count; i++)
        {
            var rows = new List<int>(imageTagRows[i].Count + 3);
            rows.AddRange(imageTagRows[i].Where(r => Array.IndexOf(cumulativeRows, r) < 0));
            for (var level = 0; level < ImageRatings.CumulativeLevel(ratings[i]); level++)
                rows.Add(cumulativeRows[level]);
            result.Add(rows);
        }

        return result;
    }
}
