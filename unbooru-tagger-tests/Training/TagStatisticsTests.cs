using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class TagStatisticsTests
{
    // Tag 0 = "character", tag 1 = "series", tag 2 = "hair color" (always on the character),
    // tag 3 = an unrelated tag. The series appears on every character image plus many
    // images without the character -- the classic hard-negative source.
    private static List<IReadOnlyList<int>> Corpus()
    {
        var images = new List<IReadOnlyList<int>>();
        for (var i = 0; i < 40; i++)
            images.Add([0, 1, 2]);
        for (var i = 0; i < 40; i++)
            images.Add([1]);
        for (var i = 0; i < 40; i++)
            images.Add([3]);
        return images;
    }

    [Fact]
    public void Build_FindsImpliedTagsAndHardNegativeSources()
    {
        var statistics = TagStatistics.Build(Corpus(), vocabularySize: 5, samplesPerTag: 1000);

        Assert.Equal(40, statistics.Positives[0].Length);
        Assert.Empty(statistics.Positives[4]);
        Assert.Contains(2, statistics.ImpliedBy[0]); // character -> its hair color
        Assert.Contains(1, statistics.ImpliedBy[0]); // character -> its series
        Assert.DoesNotContain(0, statistics.ImpliedBy[1]); // series does NOT imply the character (half its images lack it)
        Assert.Contains(1, statistics.HardNegativeSources[0]); // series images without the character
        Assert.DoesNotContain(2, statistics.HardNegativeSources[0]); // hair color never appears without the character
    }

    [Fact]
    public void SampleHardNegative_NeverReturnsAPositiveOrALikelyMissingImage()
    {
        var statistics = TagStatistics.Build(Corpus(), vocabularySize: 5, samplesPerTag: 1000);
        var random = new Random(1);

        for (var i = 0; i < 500; i++)
        {
            var image = statistics.SampleHardNegative(0, random);
            Assert.NotNull(image);
            Assert.False(statistics.HasTag(image.Value, 0));
            Assert.False(statistics.IsLikelyMissing(image.Value, 0));
        }
    }

    [Fact]
    public void TagCentricBatchSampler_PairsEachAnchorsPositivesWithItsHardNegatives()
    {
        var statistics = TagStatistics.Build(Corpus(), vocabularySize: 5, samplesPerTag: 1000);
        var sampler = new TagCentricBatchSampler(statistics, positivesPerAnchor: 2, hardNegativesPerAnchor: 2, random: new Random(3));

        Assert.Equal(4, sampler.AnchorTagCount); // tag 4 has no images

        var batch = sampler.SampleBatch(400);
        Assert.Equal(400, batch.Length);
        foreach (var sampled in batch)
            Assert.Equal(!sampled.IsHardNegative, statistics.HasTag(sampled.Image, sampled.AnchorTag));
    }
}
