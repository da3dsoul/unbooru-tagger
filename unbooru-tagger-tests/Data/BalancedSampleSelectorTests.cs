using UnbooruTagger.Data;

namespace UnbooruTagger.Tests.Data;

public class BalancedSampleSelectorTests
{
    [Fact]
    public void Select_ReturnsEqualSizedPositiveAndNegativeSets()
    {
        var matching = Enumerable.Range(0, 10).ToList();
        var zeroOverlap = Enumerable.Range(100, 50).ToList();

        var (selectedPositives, selectedNegatives) = BalancedSampleSelector.Select(matching, zeroOverlap, maxPerClass: null, new Random(1));

        Assert.Equal(10, selectedPositives.Count);
        Assert.Equal(10, selectedNegatives.Count);
    }

    [Fact]
    public void Select_RespectsMaxPerClassCap()
    {
        var matching = Enumerable.Range(0, 10).ToList();
        var zeroOverlap = Enumerable.Range(100, 50).ToList();

        var (selectedPositives, selectedNegatives) = BalancedSampleSelector.Select(matching, zeroOverlap, maxPerClass: 3, new Random(1));

        Assert.Equal(3, selectedPositives.Count);
        Assert.Equal(3, selectedNegatives.Count);
    }

    [Fact]
    public void Select_NegativesComeOnlyFromZeroOverlapPool_WhenNothingIsLeftOver()
    {
        // maxPerClass: null consumes every matching image as a positive, so there's no
        // leftover to mix in — negatives must fall back to the zero-overlap pool alone.
        var matching = Enumerable.Range(0, 5).ToList();
        var zeroOverlap = Enumerable.Range(100, 5).ToList();

        var (_, selectedNegatives) = BalancedSampleSelector.Select(matching, zeroOverlap, maxPerClass: null, new Random(1));

        Assert.All(selectedNegatives, id => Assert.True(id >= 100));
    }

    [Fact]
    public void Select_MixesLeftoverMatchingImagesIntoNegatives_WhenMatchingPoolExceedsCap()
    {
        // Only 2 zero-overlap candidates exist, but 3 negatives are requested — by the
        // pigeonhole principle at least one negative MUST come from the 17 leftover
        // matching images (ids 0-19 minus the 3 chosen as positives), deterministically
        // proving the "mix of overlap counts" behavior regardless of RNG draw.
        var matching = Enumerable.Range(0, 20).ToList();
        var zeroOverlap = Enumerable.Range(100, 2).ToList();

        var (selectedPositives, selectedNegatives) = BalancedSampleSelector.Select(matching, zeroOverlap, maxPerClass: 3, new Random(1));

        Assert.Equal(3, selectedPositives.Count);
        Assert.Equal(3, selectedNegatives.Count);
        Assert.Contains(selectedNegatives, id => id < 100);
    }
}
