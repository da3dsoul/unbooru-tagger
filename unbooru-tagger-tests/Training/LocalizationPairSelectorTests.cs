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

        // Rows are padded to LocalizationPairSelector.WidthMultiple with weight-0 pairs.
        Assert.Equal(new long[] { 1, LocalizationPairSelector.WidthMultiple }, pairs.TagRows.shape);
        var real = pairs.Weights.data<float>().ToArray().Select((w, k) => (w, k)).Where(x => x.w > 0).Select(x => x.k).ToArray();
        Assert.Equal(new long[] { 0, 1, 3, 4 }, real.Select(k => pairs.TagRows.data<long>()[k]).ToArray());
        Assert.Equal(new float[] { 0, 0, 1, 1 }, real.Select(k => pairs.FocalMask.data<float>()[k]).ToArray());
        Assert.Equal(1f, pairs.PositiveWeightSum);
    }

    [Fact]
    public void Select_PadsShorterImagesWithZeroWeight()
    {
        using var targets = Targets([1, 1, -1, 1, -1, -1], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1], 2, 3);
        using var pairs = LocalizationPairSelector.Select(targets, new float[6], [true, true, true], minedNegatives: 0);

        var width = LocalizationPairSelector.WidthMultiple;
        Assert.Equal(new long[] { 2, width }, pairs.TagRows.shape);
        var weights = pairs.Weights.data<float>().ToArray();
        Assert.Equal(new float[] { 1, 1, 1 }, weights.Take(3));
        Assert.Equal(new float[] { 1, 0, 0 }, weights.Skip(width).Take(3));
        Assert.Equal(4f, weights.Sum()); // every padding slot at weight 0
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

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void SelectOnDevice_AndItsTilePairs_AreTheCpuPairs(int seed)
    {
        // Random batches: every pair (tag, label, weight, focal) the CPU picks, the device picks too, and nothing else at a
        // weight above 0; same for the mosaic tiles built from them, including another tile's positive that is one of this
        // image's mined negatives.
        var random = new Random(seed);
        const int images = 4, tags = 30;
        var labels = Enumerable.Range(0, images * tags).Select(_ => random.NextDouble() < 0.2 ? 1f : -1f).ToArray();
        var weights = Enumerable.Range(0, images * tags).Select(_ => random.NextDouble() < 0.15 ? 0f : 1f + (float)random.NextDouble()).ToArray();
        var focal = labels.Select(l => l > 0 ? 0f : random.NextDouble() < 0.2 ? 0f : 1f).ToArray();
        var localizable = Enumerable.Range(0, tags).Select(_ => random.NextDouble() < 0.8).ToArray();
        var scores = Enumerable.Range(0, images * tags).Select(_ => (float)random.NextDouble()).ToArray();
        using var targets = Targets(labels, weights, focal, images, tags);
        using var scoreTensor = tensor(scores, [images, tags]);

        using var cpu = LocalizationPairSelector.Select(targets, scores, localizable, minedNegatives: 5);
        using var device = LocalizationPairSelector.SelectOnDevice(targets, targets, scoreTensor, localizable, minedNegatives: 5, out var labelled);
        Assert.Equal(Pairs(cpu), Pairs(device));
        Assert.Equal(cpu.PositiveWeightSum, device.PositiveWeightSum, 4);

        foreach (int[] tiles in new[] { new[] { 0, 1, 2, 3 }, new[] { 2, 0, 3, 1 } })
        {
            using var cpuTiles = Mosaic.TilePairs(cpu, targets, tiles);
            using var deviceTiles = Mosaic.TilePairsOnDevice(device, labelled, targets, tiles);
            Assert.Equal(Pairs(cpuTiles), Pairs(deviceTiles));
            Assert.Equal(cpuTiles.PositiveWeightSum, deviceTiles.PositiveWeightSum, 4);
        }
    }

    /// <summary>Per row of pairs: tag -> (label, weight, focal), for the pairs with weight above 0.</summary>
    private static List<Dictionary<long, (float, float, float)>> Pairs(LocalizationPairs pairs)
    {
        var (count, width) = ((int)pairs.TagRows.shape[0], (int)pairs.TagRows.shape[1]);
        var rows = pairs.TagRows.data<long>().ToArray();
        var labels = pairs.Labels.data<float>().ToArray();
        var weights = pairs.Weights.data<float>().ToArray();
        var focal = pairs.FocalMask.data<float>().ToArray();
        return Enumerable.Range(0, count)
            .Select(i => Enumerable.Range(i * width, width).Where(k => weights[k] > 0)
                .ToDictionary(k => rows[k], k => (labels[k], MathF.Round(weights[k], 4), focal[k])))
            .ToList();
    }

    [Fact]
    public void CapLabelled_KeepsEveryPositive_AndScalesTheSampledNegativesToTheirTotal()
    {
        // 10 positives and 290 certain negatives in a 300-tag image; capped at 50.
        var labels = Enumerable.Range(0, 300).Select(t => t < 10 ? 1f : -1f).ToArray();
        var pairs = Enumerable.Range(0, 300).ToList();
        var (kept, scale) = LocalizationPairSelector.CapLabelled(pairs, labels, 0, 50);
        Assert.Equal(50, kept.Count);
        Assert.All(Enumerable.Range(0, 10), t => Assert.Contains(t, kept));
        Assert.Equal(290f, kept.Count(t => labels[t] < 0) * scale, 3); // the kept negatives weigh as much as all of them did

        var (small, one) = LocalizationPairSelector.CapLabelled(pairs.Take(40).ToList(), labels, 0, 50);
        Assert.Equal(40, small.Count);
        Assert.Equal(1f, one);
    }
}
