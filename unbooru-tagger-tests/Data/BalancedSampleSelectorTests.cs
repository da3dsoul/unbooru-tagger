using UnbooruTagger.Data;

namespace UnbooruTagger.Tests.Data;

public class BalancedSampleSelectorTests
{
    [Fact]
    public void Select_ReturnsEqualSizedPositiveAndNegativeSets()
    {
        var positives = Enumerable.Range(0, 10).ToList();
        var negatives = Enumerable.Range(100, 50).ToList();

        var (selectedPositives, selectedNegatives) = BalancedSampleSelector.Select(positives, negatives, maxPerClass: null, new Random(1));

        Assert.Equal(10, selectedPositives.Count);
        Assert.Equal(10, selectedNegatives.Count);
    }

    [Fact]
    public void Select_RespectsMaxPerClassCap()
    {
        var positives = Enumerable.Range(0, 10).ToList();
        var negatives = Enumerable.Range(100, 50).ToList();

        var (selectedPositives, selectedNegatives) = BalancedSampleSelector.Select(positives, negatives, maxPerClass: 3, new Random(1));

        Assert.Equal(3, selectedPositives.Count);
        Assert.Equal(3, selectedNegatives.Count);
    }

    [Fact]
    public void Select_NegativesAreDrawnFromTheNegativeCandidatePool()
    {
        var positives = Enumerable.Range(0, 5).ToList();
        var negatives = Enumerable.Range(100, 5).ToList();

        var (_, selectedNegatives) = BalancedSampleSelector.Select(positives, negatives, maxPerClass: null, new Random(1));

        Assert.All(selectedNegatives, id => Assert.True(id >= 100));
    }
}
