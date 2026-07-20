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
                    break; // sorted descending — nothing further on this site can be eligible

                counts[tagCount.Name] = tagCount.PostCount;
                progress?.OnSiteTagCount?.Invoke(client.SiteName, counts.Count);
            }

            countsBySite[client.SiteName] = counts;
        }

        var allTagNames = countsBySite.Values.SelectMany(c => c.Keys).ToHashSet(StringComparer.Ordinal);
        var surveyedAt = DateTimeOffset.UtcNow;
        var results = new List<TagSurveyResult>(allTagNames.Count);

        foreach (var name in allTagNames)
        {
            int? danbooruCount = countsBySite.TryGetValue("danbooru", out var d) && d.TryGetValue(name, out var dc) ? dc : null;
            int? gelbooruCount = countsBySite.TryGetValue("gelbooru", out var g) && g.TryGetValue(name, out var gc) ? gc : null;
            var result = new TagSurveyResult(name, danbooruCount, gelbooruCount);
            var eligible = TagEligibility.IsEligible(result, minImages);

            await db.UpsertTagSurveyAsync(name, danbooruCount, gelbooruCount, eligible, surveyedAt, cancellationToken).ConfigureAwait(false);
            results.Add(result);
        }

        var eligibleTags = results.Where(t => TagEligibility.IsEligible(t, minImages)).ToList();
        return new TagSurveySummary(
            results.Count,
            eligibleTags.Count,
            TagEligibility.EstimateImageSlots(eligibleTags, minImages, maxImages));
    }
}
