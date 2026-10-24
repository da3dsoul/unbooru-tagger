using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class LocalizationPairSelectorTests
{
    private static PairTargets Targets(float[] labels, float[] weights, float[] focal, int images, int tags) =>
        new(
            tensor(labels, [images, tags]), tensor(weights, [images, tags]), tensor(focal, [images, tags]),
            labels.Zip(weights).Where(x => x.First > 0).Sum(x => x.Second));

    [Fact]
    public void Select_KeepsLocalizablePositivesCertainNegativesAndTheTopMinedNegatives()
    {
        // tags: 0 positive, 1 certain negative, 2/3/4 unverified negatives, 5 positive but not localizable, 6 weight 0
        using var targets = Targets(
            [1, -1, -1, -1, -1, 1, -1],
            [1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 1], 1, 7);
        float[] logits = [0, 0, 0.1f, 5f, 2f, 0, 9f];
        bool[] localizable = [true, true, true, true, true, false, true];

        using var pairs = LocalizationPairSelector.Select(targets, logits, localizable, minedNegatives: 2);

        Assert.Equal(new long[] { 0, 1, 3, 4 }, pairs.TagRows.data<long>().ToArray());
        Assert.Equal(new float[] { 0, 0, 1, 1 }, pairs.FocalMask.data<float>().ToArray());
        Assert.Equal(1f, pairs.PositiveWeightSum);
    }

    [Fact]
    public void Select_PadsShorterImagesWithZeroWeight()
    {
        using var targets = Targets([1, 1, -1, 1, -1, -1], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1], 2, 3);
        using var pairs = LocalizationPairSelector.Select(targets, new float[6], [true, true, true], minedNegatives: 0);

        Assert.Equal(new long[] { 2, 3 }, pairs.TagRows.shape);
        Assert.Equal(new float[] { 1, 1, 1, 1, 0, 0 }, pairs.Weights.data<float>().ToArray());
    }

    [Fact]
    public void ComputeLocalizedPairs_OverEveryPair_MatchesTheDenseLoss()
    {
        var generator = new Random(11);
        var spatial = tensor(Enumerable.Range(0, 2 * 3 * 2 * 2).Select(_ => (float)generator.NextDouble() - 0.5f).ToArray(), [2, 3, 2, 2], requires_grad: true);
        var tags = tensor(Enumerable.Range(0, 4 * 3).Select(_ => (float)generator.NextDouble() - 0.5f).ToArray(), [4, 3], requires_grad: true);
        float[] labels = [1, -1, -1, 1, -1, 1, -1, -1];
        using var targets = Targets(labels, [1, 2, 1, 1, 1, 1, 1, 0.5f], [0, 0, 1, 0, 1, 0, 1, 1], 2, 4);
        var mask = tensor(new float[] { 1, 1, 1, 0, 1, 1, 1, 1 }, [2, 1, 2, 2]);

        var dense = SigmoidContrastiveLoss.ComputeLocalizedChunked(
            spatial, tags, tensor(labels, [2, 4]), 0.5f, 4, 0f, mask, targets: targets, negativeGamma: 2f, negativeClip: 0.05f);

        using var pairs = LocalizationPairSelector.Select(targets, new float[8], [true, true, true, true], minedNegatives: 4);
        var sparse = SigmoidContrastiveLoss.ComputeLocalizedPairs(spatial, tags, pairs, 0.5f, mask, null, 2f, 0.05f).item<float>();

        Assert.Equal(dense, sparse, 5);
    }

    [Theory]
    [InlineData(new[] { 3f, 3f, 3f, 3f }, null, 3f)]      // high everywhere: the background is high too
    [InlineData(new[] { 3f, -3f, -3f, -3f }, null, -3f)]  // one peak: the background is low
    [InlineData(new[] { 3f, 3f, 3f, -3f }, new[] { 1f, 1f, 1f, 0f }, 3f)] // padding never counts as background
    public void BackgroundTerm_PenalizesTheLeastTagLikeLocations(float[] map, float[]? mask, float expectedBackgroundLogit)
    {
        using var spatial = tensor(map, [1, 1, 2, 2]);
        using var tags = tensor(new[] { 1f }, [1, 1]);
        using var spatialMask = mask is null ? null : tensor(mask, [1, 1, 2, 2]);
        using var pairs = new LocalizationPairs(
            tensor(new long[] { 0 }, [1, 1]), tensor(new[] { 1f }, [1, 1]), tensor(new[] { 1f }, [1, 1]), tensor(new[] { 0f }, [1, 1]), 1f);

        var without = SigmoidContrastiveLoss.ComputeLocalizedPairs(spatial, tags, pairs, 1f, spatialMask, null, 4f, 0.05f).item<float>();
        var with = SigmoidContrastiveLoss.ComputeLocalizedPairs(spatial, tags, pairs, 1f, spatialMask, null, 4f, 0.05f,
            backgroundWeight: 1f, backgroundFraction: 0.4f).item<float>();

        // softplus of the background's pooled logit; equal values pool to that value.
        var expected = MathF.Log(1 + MathF.Exp(expectedBackgroundLogit));
        Assert.Equal(expected, with - without, 3);
    }
}
