using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

public class TagEligibilityTests
{
    [Fact]
    public void IsEligible_True_WhenEitherSiteMeetsThreshold()
    {
        var onlyDanbooru = new TagSurveyResult("a", 600, 10);
        var onlyGelbooru = new TagSurveyResult("b", 10, 600);

        Assert.True(TagEligibility.IsEligible(onlyDanbooru, minImages: 500));
        Assert.True(TagEligibility.IsEligible(onlyGelbooru, minImages: 500));
    }

    [Fact]
    public void IsEligible_False_WhenNeitherSiteMeetsThreshold()
    {
        var tag = new TagSurveyResult("rare", 100, 200);

        Assert.False(TagEligibility.IsEligible(tag, minImages: 500));
    }

    [Fact]
    public void IsEligible_HandlesMissingSiteCounts()
    {
        var danbooruOnly = new TagSurveyResult("a", 600, null);

        Assert.True(TagEligibility.IsEligible(danbooruOnly, minImages: 500));
    }

    [Fact]
    public void EstimateImageSlots_CapsEachTagAtMaxImages()
    {
        var tags = new[]
        {
            new TagSurveyResult("common", 10000, null),
            new TagSurveyResult("medium", 700, null),
            new TagSurveyResult("rare", 100, null), // ineligible at minImages=500
        };

        var estimate = TagEligibility.EstimateImageSlots(tags, minImages: 500, maxImages: 1000);

        // common capped at 1000, medium capped at 700 (under the cap), rare excluded entirely.
        Assert.Equal(1000 + 700, estimate);
    }

    [Fact]
    public void EstimateImageSlots_ZeroWhenNoTagsEligible()
    {
        var tags = new[] { new TagSurveyResult("rare", 10, 20) };

        Assert.Equal(0, TagEligibility.EstimateImageSlots(tags, minImages: 500, maxImages: 1000));
    }
}
