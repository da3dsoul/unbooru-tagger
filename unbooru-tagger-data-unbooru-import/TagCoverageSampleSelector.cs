namespace UnbooruTagger.Data;

/// <summary>
/// Picks which images make it into a capped large-mode cache. A plain ImageId-ordered
/// prefix has no relationship to tag distribution, so a tight cap can silently drop a
/// tag's entire image set (e.g. everything tagged with a newer character all landing
/// past the cut-off). Instead, this spends the budget rarest-tag-first — up to
/// <c>minImagesPerTag</c> images per tag — so every known tag gets a fair shot at
/// training before the leftover budget is spent filling out the rest of the corpus.
/// Kept independent of EF Core so it's easy to unit test (mirrors <see cref="BalancedSampleSelector"/>).
/// </summary>
public static class TagCoverageSampleSelector
{
    public static IReadOnlySet<int> Select(
        IReadOnlyList<(int ImageId, IReadOnlyList<string> Tags)> images,
        int maxImages,
        int minImagesPerTag)
    {
        if (images.Count <= maxImages)
            return images.Select(i => i.ImageId).ToHashSet();

        var imagesByTag = new Dictionary<string, List<int>>();
        foreach (var (imageId, tags) in images)
            foreach (var tag in tags)
            {
                if (!imagesByTag.TryGetValue(tag, out var list))
                    imagesByTag[tag] = list = [];
                list.Add(imageId);
            }

        var selected = new HashSet<int>();
        foreach (var tag in imagesByTag.Keys.OrderBy(t => imagesByTag[t].Count))
        {
            if (selected.Count >= maxImages)
                break;

            var remainingBudget = maxImages - selected.Count;
            foreach (var imageId in imagesByTag[tag].Where(id => !selected.Contains(id)).Take(Math.Min(minImagesPerTag, remainingBudget)))
                selected.Add(imageId);
        }

        foreach (var (imageId, _) in images)
        {
            if (selected.Count >= maxImages)
                break;
            selected.Add(imageId);
        }

        return selected;
    }
}
