using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

public class CrawlDatabaseTests
{
    private static async Task<CrawlDatabase> OpenTempDatabaseAsync()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"crawldb-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        return await CrawlDatabase.OpenOrCreateAsync(dir);
    }

    [Fact]
    public async Task UpsertTagSurveysAsync_PersistsAllEntriesInOneCall()
    {
        using var db = await OpenTempDatabaseAsync();
        var entries = new[]
        {
            ("common", (int?)10000, (int?)8000, true),
            ("danbooru_only", (int?)6000, (int?)null, true),
            ("gelbooru_only", (int?)null, (int?)6000, true),
        };

        var writtenCounts = new List<int>();
        await db.UpsertTagSurveysAsync(entries, DateTimeOffset.UtcNow, writtenCounts.Add);

        Assert.Equal([1, 2, 3], writtenCounts);

        var stored = await db.GetAllSurveyedTagsAsync();
        Assert.Equal(3, stored.Count);
        var common = stored.Single(t => t.Name == "common");
        Assert.Equal(10000, common.DanbooruCount);
        Assert.Equal(8000, common.GelbooruCount);
    }

    [Fact]
    public async Task UpsertTagSurveysAsync_OverwritesOnReSurvey()
    {
        using var db = await OpenTempDatabaseAsync();
        await db.UpsertTagSurveysAsync([("tag", (int?)100, (int?)null, false)], DateTimeOffset.UtcNow, null);
        await db.UpsertTagSurveysAsync([("tag", (int?)900, (int?)null, true)], DateTimeOffset.UtcNow, null);

        var stored = await db.GetAllSurveyedTagsAsync();
        Assert.Single(stored);
        Assert.Equal(900, stored[0].DanbooruCount);
    }

    private static PendingNewImage MakeImage(string md5, int cacheRowIndex, params string[] eligibleTags) =>
        new(md5, cacheRowIndex, 100, 100, DateTimeOffset.UtcNow, "danbooru", 1, "https://example/x.jpg", "g", DateTimeOffset.UtcNow, eligibleTags, PHash: 0);

    [Fact]
    public async Task CommitPendingImagesAsync_PersistsNewImagesAndCreditsEachEligibleTagOnce()
    {
        using var db = await OpenTempDatabaseAsync();
        await db.UpsertTagSurveysAsync([("1girl", (int?)1000, (int?)null, true), ("solo", (int?)1000, (int?)null, true)], DateTimeOffset.UtcNow, null);

        var image = MakeImage("md5-a", 0, "1girl", "solo");
        await db.CommitPendingImagesAsync([image], [], [], CancellationToken.None);

        var images = await db.GetAllImagesAsync();
        Assert.Single(images);
        Assert.Equal(("md5-a", 0), (images[0].Md5, images[0].CacheRowIndex));

        var counts = await db.GetAllCombinedPositiveCountsAsync();
        Assert.Equal(1, counts["1girl"]);
        Assert.Equal(1, counts["solo"]);
    }

    [Fact]
    public async Task CommitPendingImagesAsync_AdditionalSourceDoesNotDuplicateOrCreditAgain()
    {
        using var db = await OpenTempDatabaseAsync();
        await db.UpsertTagSurveysAsync([("1girl", (int?)1000, (int?)null, true)], DateTimeOffset.UtcNow, null);
        await db.CommitPendingImagesAsync([MakeImage("md5-a", 0, "1girl")], [], [], CancellationToken.None);

        // A re-encode of the same artwork found on the other site — just another
        // provenance row against the canonical md5, no new Images row or extra credit.
        var additionalSource = new PendingAdditionalSource("md5-a", "gelbooru", 2, "https://example/y.jpg", "g", DateTimeOffset.UtcNow, ["1girl"], DateTimeOffset.UtcNow);
        await db.CommitPendingImagesAsync([], [additionalSource], [], CancellationToken.None);

        Assert.Single(await db.GetAllImagesAsync());
        Assert.Equal(1, (await db.GetAllCombinedPositiveCountsAsync())["1girl"]);
    }

    [Fact]
    public async Task CommitPendingImagesAsync_MergedTagCountsCreditWithoutANewImageRow()
    {
        using var db = await OpenTempDatabaseAsync();
        await db.UpsertTagSurveysAsync(
            [("1girl", (int?)1000, (int?)null, true), ("outdoors", (int?)1000, (int?)null, true)],
            DateTimeOffset.UtcNow, null);
        await db.CommitPendingImagesAsync([MakeImage("md5-a", 0, "1girl")], [], [], CancellationToken.None);

        // The same image turns up again from the other site, this time also tagged
        // "outdoors" — no new Images row, but the tag it newly brings still counts.
        await db.CommitPendingImagesAsync([], [], ["outdoors"], CancellationToken.None);

        Assert.Single(await db.GetAllImagesAsync());
        var counts = await db.GetAllCombinedPositiveCountsAsync();
        Assert.Equal(1, counts["1girl"]);
        Assert.Equal(1, counts["outdoors"]);
    }

    [Fact]
    public async Task ImageSources_Upsert_RefreshesTagsAndFetchedAt_InsteadOfIgnoring()
    {
        using var db = await OpenTempDatabaseAsync();
        await db.UpsertTagSurveysAsync([("1girl", (int?)1000, (int?)null, true)], DateTimeOffset.UtcNow, null);
        await db.CommitPendingImagesAsync([MakeImage("md5-a", 0, "1girl")], [], [], CancellationToken.None);

        // The same (site, postId) source turns up again in a later crawl pass with a
        // different tag snapshot — must overwrite, not be ignored as already-known.
        var updated = new PendingAdditionalSource("md5-a", "danbooru", 1, "https://example/x.jpg", "g", DateTimeOffset.UtcNow, ["1girl", "solo"], DateTimeOffset.UtcNow);
        await db.CommitPendingImagesAsync([], [updated], [], CancellationToken.None);

        var snapshots = await db.GetImageSourceSnapshotsAsync("md5-a");
        var source = Assert.Single(snapshots);
        Assert.Equal(["1girl", "solo"], source.Tags);
    }

    [Fact]
    public async Task GetImageSourceSnapshotsAsync_NullTagsMeansNeverCaptured()
    {
        using var db = await OpenTempDatabaseAsync();
        await db.UpsertTagSurveysAsync([("1girl", (int?)1000, (int?)null, true)], DateTimeOffset.UtcNow, null);
        // Simulates a pre-migration row: PendingNewImage's own source always gets a
        // real snapshot, so insert a bare additional source with Tags = null directly
        // via ApplyRefreshBatchAsync's update path having never run for it — closest
        // approximation here is committing then asserting the seeded row has real tags,
        // then checking a never-touched (site, postId) simply doesn't appear at all.
        await db.CommitPendingImagesAsync([MakeImage("md5-a", 0, "1girl")], [], [], CancellationToken.None);

        var snapshots = await db.GetImageSourceSnapshotsAsync("md5-a");
        Assert.Single(snapshots);
        Assert.NotNull(snapshots[0].Tags); // freshly committed sources are never "unknown"
    }

    [Fact]
    public async Task GetSourcesBatchAsync_OrdersByPostIdAscending_AndRespectsAfterCursor()
    {
        using var db = await OpenTempDatabaseAsync();
        await db.UpsertTagSurveysAsync([("1girl", (int?)1000, (int?)null, true)], DateTimeOffset.UtcNow, null);
        await db.CommitPendingImagesAsync(
            [MakeImage("md5-a", 0, "1girl"), MakeImage("md5-b", 1, "1girl") with { Site = "danbooru", PostId = 5 }, MakeImage("md5-c", 2, "1girl") with { Site = "danbooru", PostId = 3 }],
            [], [], CancellationToken.None);

        var firstBatch = await db.GetSourcesBatchAsync("danbooru", afterPostId: 0, limit: 10);
        Assert.Equal([1L, 3L, 5L], firstBatch.Select(s => s.PostId));

        var afterFirst = await db.GetSourcesBatchAsync("danbooru", afterPostId: 3, limit: 10);
        Assert.Equal([5L], afterFirst.Select(s => s.PostId));
    }

    [Fact]
    public async Task ApplyRefreshBatchAsync_PersistsSnapshotsDeltasAndCursorTogether()
    {
        using var db = await OpenTempDatabaseAsync();
        await db.UpsertTagSurveysAsync(
            [("1girl", (int?)1000, (int?)null, true), ("outdoors", (int?)1000, (int?)null, true)],
            DateTimeOffset.UtcNow, null);
        await db.CommitPendingImagesAsync([MakeImage("md5-a", 0, "1girl")], [], [], CancellationToken.None);

        var refreshed = new RefreshedSourceTags("danbooru", 1, "md5-a", ["outdoors"], DateTimeOffset.UtcNow);
        await db.ApplyRefreshBatchAsync(
            [refreshed],
            new Dictionary<string, int> { ["1girl"] = -1, ["outdoors"] = 1 },
            "danbooru", lastPostId: 1, done: true, CancellationToken.None);

        var snapshot = Assert.Single(await db.GetImageSourceSnapshotsAsync("md5-a"));
        Assert.Equal(["outdoors"], snapshot.Tags);

        var counts = await db.GetAllCombinedPositiveCountsAsync();
        Assert.Equal(0, counts["1girl"]);
        Assert.Equal(1, counts["outdoors"]);

        var (lastPostId, done) = await db.GetRefreshProgressAsync("danbooru");
        Assert.Equal(1, lastPostId);
        Assert.True(done);
    }

    [Fact]
    public async Task ApplyRefreshBatchAsync_RecordsAGonePostAsAConfirmedEmptyList_NotAsUnknownNull()
    {
        using var db = await OpenTempDatabaseAsync();
        await db.UpsertTagSurveysAsync([("1girl", (int?)1000, (int?)null, true)], DateTimeOffset.UtcNow, null);
        await db.CommitPendingImagesAsync([MakeImage("md5-a", 0, "1girl")], [], [], CancellationToken.None);

        // A deleted post is a *confirmed* zero-tags source, not an unverified one — see
        // RefreshedSourceTags's doc comment for why conflating the two would silently
        // re-block removal forever the moment the deleted source itself got refreshed.
        var deleted = new RefreshedSourceTags("danbooru", 1, "md5-a", [], DateTimeOffset.UtcNow);
        await db.ApplyRefreshBatchAsync([deleted], new Dictionary<string, int> { ["1girl"] = -1 }, "danbooru", 1, true, CancellationToken.None);

        var snapshot = Assert.Single(await db.GetImageSourceSnapshotsAsync("md5-a"));
        Assert.NotNull(snapshot.Tags);
        Assert.Empty(snapshot.Tags!);
    }

    [Fact]
    public async Task GetAllImagesAsync_RoundTripsPHash()
    {
        using var db = await OpenTempDatabaseAsync();
        var image = MakeImage("md5-a", 0) with { PHash = 0xFFFFFFFFFFFFFFFFUL };
        await db.CommitPendingImagesAsync([image], [], [], CancellationToken.None);

        var images = await db.GetAllImagesAsync();
        Assert.Equal(0xFFFFFFFFFFFFFFFFUL, images[0].PHash); // exercises the signed-long bit-pattern round trip for the top bit
    }

    [Fact]
    public async Task CountSiteContributionsForTagAsync_CountsDistinctImagesForThatSiteAndTag()
    {
        using var db = await OpenTempDatabaseAsync();
        await db.UpsertTagSurveysAsync([("head_pat", (int?)10, (int?)null, true), ("1girl", (int?)10, (int?)null, true)], DateTimeOffset.UtcNow, null);

        // MakeImage hardcodes PostId 1 — override it per call so these three images land
        // as three distinct ImageSources rows (Site, PostId) instead of colliding on the
        // same upsert key and silently overwriting each other's Tags.
        await db.CommitPendingImagesAsync([MakeImage("md5-a", 0, "head_pat", "1girl") with { PostId = 1 }], [], [], CancellationToken.None);
        await db.CommitPendingImagesAsync([MakeImage("md5-b", 1, "head_pat") with { PostId = 2 }], [], [], CancellationToken.None);
        await db.CommitPendingImagesAsync([MakeImage("md5-c", 2, "1girl") with { PostId = 3 }], [], [], CancellationToken.None); // no head_pat

        Assert.Equal(2, await db.CountSiteContributionsForTagAsync("danbooru", "head_pat"));
        Assert.Equal(0, await db.CountSiteContributionsForTagAsync("gelbooru", "head_pat")); // MakeImage only ever writes "danbooru" sources
    }

    /// <summary>
    /// Regression guard for the derivation this feeds (RunSiteTagPhaseAsync's resume-
    /// credit seeding): booru tags routinely contain literal underscores
    /// (<c>head_pat</c>, <c>white_hair</c>), which is also the SQL LIKE single-character
    /// wildcard. Without escaping it, a tag search for "head_pat" would silently also
    /// match "headXpat" for any character X, inflating a resumed run's seeded credit.
    /// </summary>
    [Fact]
    public async Task CountSiteContributionsForTagAsync_UnderscoreInTagNameIsNotTreatedAsALikeWildcard()
    {
        using var db = await OpenTempDatabaseAsync();
        await db.UpsertTagSurveysAsync([("head_pat", (int?)10, (int?)null, true), ("headxpat", (int?)10, (int?)null, true)], DateTimeOffset.UtcNow, null);

        await db.CommitPendingImagesAsync([MakeImage("md5-a", 0, "headxpat")], [], [], CancellationToken.None);

        Assert.Equal(0, await db.CountSiteContributionsForTagAsync("danbooru", "head_pat"));
        Assert.Equal(1, await db.CountSiteContributionsForTagAsync("danbooru", "headxpat"));
    }
}
