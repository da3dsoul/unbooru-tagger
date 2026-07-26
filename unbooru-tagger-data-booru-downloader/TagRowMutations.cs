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

    /// <summary>
    /// Maps a post's raw (un-prefixed) tag names — the only form a site's API ever
    /// returns — to their eligible <see cref="TagCategoryNaming"/> identities, dropping
    /// any raw tag that isn't a currently-eligible tag at all. <paramref name="eligibleTagIdentities"/>
    /// is keyed by raw name (see <see cref="BuildEligibleIdentities"/> for how it's built
    /// from the survey) since that's the only form a post's tags ever arrive in.
    /// </summary>
    public static IEnumerable<string> EligibleIdentities(IEnumerable<string> rawTags, IReadOnlyDictionary<string, string> eligibleTagIdentities) =>
        rawTags.Select(eligibleTagIdentities.GetValueOrDefault).OfType<string>();

    /// <summary>
    /// Builds the raw-name -&gt; identity lookup <see cref="EligibleIdentities"/> consumes.
    /// Deduped by raw name — not just grouped by identity — because two sites can (and do)
    /// disagree on a tag's category (e.g. Danbooru survey files "elvaan" as General while
    /// Gelbooru calls it Character), which <c>TagSurveyor</c> now reconciles into a single
    /// identity going forward, but a <c>crawl.sqlite</c> surveyed before that fix can still
    /// hold both stale rows. Rather than crash on the collision, keep the identity with the
    /// higher combined post count (ties broken by identity string for determinism).
    /// </summary>
    /// <param name="tagAliases">
    /// Antecedent raw name -&gt; consequent raw name (see
    /// <see cref="DanbooruClient.ListActiveTagAliasesAsync"/>). <c>TagSurveyor</c> folds a
    /// known antecedent into its consequent before a tag ever becomes its own eligible
    /// survey row, which means the antecedent raw name is otherwise MISSING from
    /// <paramref name="eligibleTags"/> entirely — not just merged, gone. Without this
    /// parameter, a post still carrying the antecedent spelling (e.g. Gelbooru's own posts
    /// keep saying <c>head_pat</c> forever; Gelbooru has no idea Danbooru aliased it to
    /// <c>headpat</c>) would silently drop that tag altogether instead of crediting it to
    /// the merged identity — worse than before the alias merge, which at least recorded it
    /// under the wrong name. This backfills every known antecedent as an extra key
    /// pointing at its (possibly chain-resolved) consequent's real eligible identity, so
    /// that gap never happens.
    /// </param>
    public static Dictionary<string, string> BuildEligibleIdentities(
        IEnumerable<TagSurveyResult> eligibleTags,
        IReadOnlyDictionary<string, string>? tagAliases = null)
    {
        var byRawName = eligibleTags
            .GroupBy(t => TagCategoryNaming.RawName(t.Name), StringComparer.Ordinal)
            .ToDictionary(
                g => g.Key,
                g => g.OrderByDescending(t => t.BestCount).ThenBy(t => t.Name, StringComparer.Ordinal).First().Name,
                StringComparer.Ordinal);

        if (tagAliases is { Count: > 0 })
        {
            foreach (var antecedent in tagAliases.Keys)
            {
                if (byRawName.ContainsKey(antecedent))
                    continue; // a real (non-aliased) eligible tag already owns this raw name

                var canonicalRawName = ResolveAlias(antecedent, tagAliases);
                if (byRawName.TryGetValue(canonicalRawName, out var identity))
                    byRawName[antecedent] = identity;
            }
        }

        return byRawName;
    }

    /// <summary>Follows <paramref name="tagAliases"/> (antecedent raw name -&gt; consequent raw name) from <paramref name="rawName"/> to its final target, cycle-safe in case of a bad/circular alias chain.</summary>
    public static string ResolveAlias(string rawName, IReadOnlyDictionary<string, string> tagAliases)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var current = rawName;
        while (tagAliases.TryGetValue(current, out var next) && seen.Add(current))
            current = next;
        return current;
    }
}
