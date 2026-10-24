using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class PairWeightBuilderTests
{
    // 0 solo, 1 multiple girls, 2 character, 3 its hair color (implied), 4 unrelated, 5 never seen.
    private static (TagStatistics Statistics, ExclusiveTagGroups Groups) Setup()
    {
        var images = new List<IReadOnlyList<int>>();
        for (var i = 0; i < 20; i++)
            images.Add([0, 2, 3]);
        for (var i = 0; i < 20; i++)
            images.Add([1, 4]);

        var vocabulary = TagVocabulary.CreateEmpty();
        foreach (var tag in new[] { "solo", "multiple girls", "character", "hair", "other", "stale" })
            vocabulary.AddTag(tag);

        return (TagStatistics.Build(images, 6, samplesPerTag: 1000), ExclusiveTagGroups.Parse("solo, multiple_girls", vocabulary));
    }

    [Fact]
    public void Build_AssignsEachPairItsCategory()
    {
        var (statistics, groups) = Setup();
        var builder = new PairWeightBuilder(statistics, groups, certainNegativeWeight: 2f);

        // An image tagged solo + character but missing the hair tag, drawn as a hard
        // negative for "other".
        using var targets = builder.Build([[0, 2]], [new SampledImage(0, AnchorTag: 4, IsHardNegative: true)]);
        var labels = targets.Labels.data<float>().ToArray();
        var weights = targets.Weights.data<float>().ToArray();
        var focal = targets.FocalMask.data<float>().ToArray();

        Assert.Equal((1f, 1f, 0f), (labels[0], weights[0], focal[0]));   // positive
        Assert.Equal((-1f, 2f, 0f), (labels[1], weights[1], focal[1]));  // exclusive with solo: certain
        Assert.Equal((1f, 1f, 0f), (labels[2], weights[2], focal[2]));   // positive
        Assert.Equal(0f, weights[3]);                                     // implied by character: likely missing
        Assert.Equal((-1f, 2f, 0f), (labels[4], weights[4], focal[4]));  // anchor hard negative: certain
        Assert.Equal(0f, weights[5]);                                     // no training images at all
        Assert.Equal(2f, targets.PositiveWeightSum);
    }

    [Fact]
    public void Build_UnverifiedNegativesAreFocalDamped()
    {
        var (statistics, groups) = Setup();
        var builder = new PairWeightBuilder(statistics, groups);

        using var targets = builder.Build([[1]]);
        var weights = targets.Weights.data<float>().ToArray();
        var focal = targets.FocalMask.data<float>().ToArray();

        Assert.Equal((1f, 1f), (weights[2], focal[2])); // character: absent, nothing else known
    }

    [Fact]
    public void Build_ExhaustiveRowsAreCertainNegativesEvenWhenImplied()
    {
        var (statistics, groups) = Setup();
        // Row 3 is implied by the character (row 2); as an exhaustively labeled row (like a
        // rating) its absence is still a verified "no".
        var builder = new PairWeightBuilder(statistics, groups, certainNegativeWeight: 2f, exhaustiveRows: [3]);

        using var targets = builder.Build([[0, 2], [0, 2, 3]]);
        var labels = targets.Labels.data<float>().ToArray();
        var weights = targets.Weights.data<float>().ToArray();
        var focal = targets.FocalMask.data<float>().ToArray();

        Assert.Equal((-1f, 2f, 0f), (labels[3], weights[3], focal[3]));
        Assert.Equal((1f, 1f, 0f), (labels[6 + 3], weights[6 + 3], focal[6 + 3]));
    }
}
