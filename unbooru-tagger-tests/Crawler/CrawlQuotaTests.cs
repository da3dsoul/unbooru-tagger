using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

public class CrawlQuotaTests
{
    [Theory]
    [InlineData(0, 1000, true)]
    [InlineData(999, 1000, true)]
    [InlineData(1000, 1000, false)]
    [InlineData(1500, 1000, false)]
    public void ShouldContinueFetching_StopsOnceQuotaMet(int combinedCountSoFar, int maxImages, bool expected)
    {
        Assert.Equal(expected, CrawlQuota.ShouldContinueFetching(combinedCountSoFar, maxImages));
    }

    [Fact]
    public void NeedsDownload_FalseForKnownMd5_TrueForUnknown()
    {
        Assert.False(CrawlQuota.NeedsDownload(md5AlreadyKnown: true));
        Assert.True(CrawlQuota.NeedsDownload(md5AlreadyKnown: false));
    }

    [Fact]
    public void NegativeShortfall_PositiveWhenNegativePoolTooSmall()
    {
        // 1000 total images, 950 of them positive for this tag -> only 50 negatives locally.
        var shortfall = CrawlQuota.NegativeShortfall(totalImages: 1000, combinedPositiveCount: 950, negativeTarget: 1000);

        Assert.Equal(950, shortfall); // needs 1000 negatives, has 50 -> short by 950
    }

    [Fact]
    public void NegativeShortfall_ZeroOnceTargetAlreadyMet()
    {
        // 1000 total images, only 100 positive for this tag -> 900 negatives already, target 500.
        var shortfall = CrawlQuota.NegativeShortfall(totalImages: 1000, combinedPositiveCount: 100, negativeTarget: 500);

        Assert.Equal(0, shortfall);
    }

    [Fact]
    public void NegativeShortfall_NeverNegative()
    {
        var shortfall = CrawlQuota.NegativeShortfall(totalImages: 100, combinedPositiveCount: 0, negativeTarget: 10);

        Assert.True(shortfall >= 0);
        Assert.Equal(0, shortfall);
    }
}

public class CrawlSchedulingTests
{
    [Fact]
    public void RarestFirst_OrdersAscendingByBestCount()
    {
        var tags = new[]
        {
            new TagSurveyResult("common", 10000, null),
            new TagSurveyResult("rare", 501, null),
            new TagSurveyResult("medium", 700, null),
        };

        var ordered = CrawlScheduling.RarestFirst(tags).Select(t => t.Name).ToList();

        Assert.Equal(["rare", "medium", "common"], ordered);
    }

    [Fact]
    public void RarestFirst_BreaksTiesByNameForDeterminism()
    {
        var tags = new[]
        {
            new TagSurveyResult("zebra", 500, null),
            new TagSurveyResult("apple", 500, null),
        };

        var ordered = CrawlScheduling.RarestFirst(tags).Select(t => t.Name).ToList();

        Assert.Equal(["apple", "zebra"], ordered);
    }

    [Fact]
    public void PickLeastLoadedSite_PicksFewestRequestsSoFar()
    {
        var requests = new Dictionary<string, int> { ["danbooru"] = 10, ["gelbooru"] = 3 };

        Assert.Equal("gelbooru", CrawlScheduling.PickLeastLoadedSite(requests));
    }

    [Fact]
    public void PickLeastLoadedSite_BreaksTiesByNameForDeterminism()
    {
        var requests = new Dictionary<string, int> { ["danbooru"] = 5, ["gelbooru"] = 5 };

        Assert.Equal("danbooru", CrawlScheduling.PickLeastLoadedSite(requests));
    }

    [Fact]
    public void PickLeastLoadedSite_ThrowsWhenEmpty()
    {
        Assert.Throws<ArgumentException>(() => CrawlScheduling.PickLeastLoadedSite(new Dictionary<string, int>()));
    }
}
