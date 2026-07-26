namespace UnbooruTagger.Crawler;

/// <summary>Summary printed at the end of <c>survey-tags</c> and again as the first step of <c>crawl</c> — see CLAUDE.md-adjacent plan notes on up-front estimation.</summary>
public sealed record TagSurveySummary(
    int TotalTagsSeen,
    int EligibleTagCount,
    long EstimatedImageSlots,
    int ExcludedTagCount);

/// <summary>Reports what <see cref="TagSurveyor"/> is doing right now, for a caller to render as a live status line.</summary>
public sealed class TagSurveyProgress
{
    public Action<string, int>? OnSiteTagCount { get; init; }

    /// <summary>
    /// Fired once per site, exactly when the descending scan hits a tag below
    /// <c>--min-images</c> and stops. Surfaces the tag/count that triggered the stop so
    /// a caller can sanity-check the site's API is actually honoring the
    /// sort-by-count-descending request — if the count found so far is tiny relative to
    /// what the site's tag vocabulary is known to be, that's a sign the site silently
    /// ignored the sort and this stopped far too early, rather than a case of the site
    /// genuinely having few eligible tags.
    /// </summary>
    public Action<string, int, string, int>? OnSiteStopped { get; init; }

    /// <summary>Fired while persisting survey results to <c>crawl.sqlite</c> — the fetch phase can look done while this, a separate and previously invisible phase, is still writing.</summary>
    public Action<int, int>? OnPersisting { get; init; }
}

/// <summary>
/// Implements <c>survey-tags</c>: for each site, pages the tags-list endpoint sorted by
/// post count descending and stops as soon as counts drop below <paramref name="minImages"/>
/// — cheap, since the ordering means nothing past that point could be eligible.
/// </summary>
public static class TagSurveyor
{
    /// <param name="tagAliases">
    /// Antecedent raw name -&gt; consequent raw name (see <see cref="DanbooruClient.ListActiveTagAliasesAsync"/>).
    /// Folded into both sites' raw-name counts before grouping, so a raw name one site
    /// still reports under a deprecated spelling (e.g. Gelbooru's <c>head_pat</c>, where
    /// Danbooru's own current tag is <c>headpat</c>) is treated as the SAME tag as its
    /// alias target instead of surveyed as its own, unrelated eligible tag. Without this,
    /// a site whose live search silently redirects an aliased query returns posts tagged
    /// with the consequent name, which can never satisfy quota tracked against the
    /// antecedent identity — see <see cref="DanbooruClient.ListActiveTagAliasesAsync"/>'s
    /// doc comment for the full failure mode. Any existing survey row for a known
    /// antecedent is deleted outright (not just marked ineligible) once this runs, so a
    /// crawl.sqlite surveyed before this alias table was wired in doesn't keep re-crawling
    /// the stale identity forever.
    /// </param>
    public static async Task<TagSurveySummary> SurveyAsync(
        CrawlDatabase db,
        IReadOnlyList<IBooruClient> clients,
        int minImages,
        int maxImages,
        TagExclusionRules? excludedTags = null,
        IReadOnlyDictionary<string, string>? tagAliases = null,
        TagSurveyProgress? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Site -> (tag -> count), merged into one per-tag record per site below.
        var countsBySite = new Dictionary<string, Dictionary<string, int>>();

        foreach (var client in clients)
        {
            var counts = new Dictionary<string, int>(StringComparer.Ordinal);
            await foreach (var tagCount in client.ListTagsByCountDescendingAsync(cancellationToken).WithCancellation(cancellationToken))
            {
                if (tagCount.PostCount < minImages)
                {
                    // sorted descending — nothing further on this site can be eligible
                    progress?.OnSiteStopped?.Invoke(client.SiteName, counts.Count, tagCount.Name, tagCount.PostCount);
                    break;
                }

                counts[tagCount.Name] = tagCount.PostCount;
                progress?.OnSiteTagCount?.Invoke(client.SiteName, counts.Count);
            }

            countsBySite[client.SiteName] = counts;
        }

        // Merged by raw name, not by identity string: sites categorize the same raw tag
        // differently often enough (e.g. Danbooru files a tag as General while Gelbooru
        // calls it Character) that merging by identity would keep both as separate rows —
        // which downstream code can't tolerate, since everything past this point (vocab,
        // exclusions, TagRowMutations.EligibleIdentities) assumes a raw name maps to
        // exactly one identity. Danbooru's categorization wins on disagreement since it's
        // looked up first below and is generally the better-curated of the two.
        Dictionary<string, (TagCategory Category, int Count)> ByRawName(string site)
        {
            var byRaw = new Dictionary<string, (TagCategory, int)>(StringComparer.Ordinal);
            foreach (var (identity, count) in countsBySite.GetValueOrDefault(site) ?? [])
            {
                var (rawName, category) = TagCategoryNaming.Split(identity);
                byRaw[rawName] = (category, count);
            }
            return byRaw;
        }

        // Folds a known alias antecedent into its consequent before grouping — see this
        // method's own doc comment on tagAliases for why (head_pat/headpat, mind_break/
        // mindbreak, ...). Applied to both sites' raw names, not just whichever site the
        // alias table came from: the antecedent spelling can show up as the CURRENT raw
        // name on the other site even though the alias itself is only known from
        // Danbooru's own table.
        Dictionary<string, (TagCategory Category, int Count)> ResolveAliases(Dictionary<string, (TagCategory Category, int Count)> byRaw)
        {
            if (tagAliases is not { Count: > 0 })
                return byRaw;

            var resolved = new Dictionary<string, (TagCategory Category, int Count)>(StringComparer.Ordinal);
            foreach (var (rawName, value) in byRaw)
            {
                var canonical = TagRowMutations.ResolveAlias(rawName, tagAliases);
                resolved[canonical] = resolved.TryGetValue(canonical, out var existing)
                    ? (existing.Category, existing.Count + value.Count)
                    : value;
            }
            return resolved;
        }

        var danbooruByRaw = ResolveAliases(ByRawName("danbooru"));
        var gelbooruByRaw = ResolveAliases(ByRawName("gelbooru"));
        var rawNames = danbooruByRaw.Keys.Concat(gelbooruByRaw.Keys).ToHashSet(StringComparer.Ordinal);

        var surveyedAt = DateTimeOffset.UtcNow;
        var results = new List<TagSurveyResult>(rawNames.Count);
        var entries = new List<(string Name, int? DanbooruCount, int? GelbooruCount, bool Eligible)>(rawNames.Count);

        foreach (var rawName in rawNames)
        {
            var hasDanbooru = danbooruByRaw.TryGetValue(rawName, out var danbooru);
            var hasGelbooru = gelbooruByRaw.TryGetValue(rawName, out var gelbooru);
            int? danbooruCount = hasDanbooru ? danbooru.Count : null;
            int? gelbooruCount = hasGelbooru ? gelbooru.Count : null;

            var category = hasDanbooru ? danbooru.Category : hasGelbooru ? gelbooru.Category : TagCategory.General;
            var identity = TagCategoryNaming.Identity(rawName, category);

            var result = new TagSurveyResult(identity, danbooruCount, gelbooruCount);
            var eligible = TagEligibility.IsEligible(result, minImages, excludedTags);

            entries.Add((identity, danbooruCount, gelbooruCount, eligible));
            results.Add(result);
        }

        // One transaction for every row instead of one auto-committed write each — with
        // tens of thousands of eligible tags, per-row commits (each its own fsync) were
        // the dominant cost of this command and gave no visible progress at all.
        await db.UpsertTagSurveysAsync(
            entries,
            surveyedAt,
            written => progress?.OnPersisting?.Invoke(written, entries.Count),
            cancellationToken).ConfigureAwait(false);

        // A known antecedent never survives ResolveAliases into `entries` above (it's
        // folded into its consequent before rawNames is even built), so any row still
        // sitting in the DB under that identity is left over from before this alias was
        // known — e.g. a crawl.sqlite surveyed before ListActiveTagAliasesAsync existed.
        // Deleted outright rather than left at Eligible = 0, since that path is for a tag
        // that genuinely fell under quota this survey, not one that should never be
        // iterated as its own tag again.
        if (tagAliases is { Count: > 0 })
        {
            var existingRows = await db.GetAllSurveyedTagsAsync(cancellationToken).ConfigureAwait(false);
            var staleAliasRows = existingRows
                .Select(t => t.Name)
                .Where(name => tagAliases.ContainsKey(TagCategoryNaming.RawName(name)))
                .ToList();
            if (staleAliasRows.Count > 0)
                await db.DeleteTagSurveysAsync(staleAliasRows, cancellationToken).ConfigureAwait(false);
        }

        var eligibleTags = results.Where(t => TagEligibility.IsEligible(t, minImages, excludedTags)).ToList();
        var excludedTagCount = excludedTags is null ? 0 : results.Count(t => excludedTags.IsExcluded(t.Name));
        return new TagSurveySummary(
            results.Count,
            eligibleTags.Count,
            TagEligibility.EstimateImageSlots(eligibleTags, minImages, maxImages, excludedTags),
            excludedTagCount);
    }
}
