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
        await db.CommitPendingImagesAsync([image], [], CancellationToken.None);

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
        await db.CommitPendingImagesAsync([MakeImage("md5-a", 0, "1girl")], [], CancellationToken.None);

        // A re-encode of the same artwork found on the other site — just another
        // provenance row against the canonical md5, no new Images row or extra credit.
        var additionalSource = new PendingAdditionalSource("md5-a", "gelbooru", 2, "https://example/y.jpg", "g", DateTimeOffset.UtcNow);
        await db.CommitPendingImagesAsync([], [additionalSource], CancellationToken.None);

        Assert.Single(await db.GetAllImagesAsync());
        Assert.Equal(1, (await db.GetAllCombinedPositiveCountsAsync())["1girl"]);
    }

    [Fact]
    public async Task GetAllImagesAsync_RoundTripsPHash()
    {
        using var db = await OpenTempDatabaseAsync();
        var image = MakeImage("md5-a", 0) with { PHash = 0xFFFFFFFFFFFFFFFFUL };
        await db.CommitPendingImagesAsync([image], [], CancellationToken.None);

        var images = await db.GetAllImagesAsync();
        Assert.Equal(0xFFFFFFFFFFFFFFFFUL, images[0].PHash); // exercises the signed-long bit-pattern round trip for the top bit
    }
}
