namespace UnbooruTagger.Crawler;

/// <summary>Summary printed at the end of <c>survey-tags</c> and again as the first step of <c>crawl</c> — see CLAUDE.md-adjacent plan notes on up-front estimation.</summary>
public sealed record TagSurveySummary(
    int TotalTagsSeen,
    int EligibleTagCount,
    long EstimatedImageSlots);

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
    public static async Task<TagSurveySummary> SurveyAsync(
        CrawlDatabase db,
        IReadOnlyList<IBooruClient> clients,
        int minImages,
        int maxImages,
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

        var allTagNames = countsBySite.Values.SelectMany(c => c.Keys).ToHashSet(StringComparer.Ordinal);
        var surveyedAt = DateTimeOffset.UtcNow;
        var results = new List<TagSurveyResult>(allTagNames.Count);
        var entries = new List<(string Name, int? DanbooruCount, int? GelbooruCount, bool Eligible)>(allTagNames.Count);

        foreach (var name in allTagNames)
        {
            int? danbooruCount = countsBySite.TryGetValue("danbooru", out var d) && d.TryGetValue(name, out var dc) ? dc : null;
            int? gelbooruCount = countsBySite.TryGetValue("gelbooru", out var g) && g.TryGetValue(name, out var gc) ? gc : null;
            var result = new TagSurveyResult(name, danbooruCount, gelbooruCount);
            var eligible = TagEligibility.IsEligible(result, minImages);

            entries.Add((name, danbooruCount, gelbooruCount, eligible));
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

        var eligibleTags = results.Where(t => TagEligibility.IsEligible(t, minImages)).ToList();
        return new TagSurveySummary(
            results.Count,
            eligibleTags.Count,
            TagEligibility.EstimateImageSlots(eligibleTags, minImages, maxImages));
    }
}
