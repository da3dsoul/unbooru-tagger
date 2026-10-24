using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class SigmoidContrastiveLossTests
{
    [Fact]
    public void Compute_PenalizesMismatchMoreThanMatch()
    {
        var imageEmbedding = tensor(new float[] { 1, 0 }, [1, 2]);
        var matchingTag = tensor(new float[] { 1, 0 }, [1, 2]);
        var mismatchedTag = tensor(new float[] { -1, 0 }, [1, 2]);
        var labels = tensor(new float[] { 1 }, [1, 1]);

        var lossForMatch = SigmoidContrastiveLoss.Compute(imageEmbedding, matchingTag, labels).item<float>();
        var lossForMismatch = SigmoidContrastiveLoss.Compute(imageEmbedding, mismatchedTag, labels).item<float>();

        Assert.True(lossForMatch < lossForMismatch);
    }

    [Fact]
    public void Compute_WithScaling_MatchesApplyingItToTheRawDotProductDirectly()
    {
        var imageEmbedding = tensor(new float[] { 1, 0 }, [1, 2]);
        var tagEmbedding = tensor(new float[] { 1, 0 }, [1, 2]);
        var labels = tensor(new float[] { 1 }, [1, 1]);
        var scaling = new LogitScaleBias(initialScale: 3f, initialBias: 1f);

        // dot(imageEmbedding, tagEmbedding) == 1, so the scaled logit is 3*1 + 1 == 4 --
        // clearly different from the raw dot product, so this can't coincidentally match
        // the unscaled loss below by accident.
        var expected = SigmoidContrastiveLoss.Compute(imageEmbedding, tagEmbedding, labels).item<float>();
        var withIdentityScaling = SigmoidContrastiveLoss.Compute(imageEmbedding, tagEmbedding, labels, new LogitScaleBias(initialScale: 1f, initialBias: 0f)).item<float>();
        var withRealScaling = SigmoidContrastiveLoss.Compute(imageEmbedding, tagEmbedding, labels, scaling).item<float>();

        Assert.Equal(expected, withIdentityScaling, 4);
        Assert.NotEqual(expected, withRealScaling, 3);
    }

    [Fact]
    public void ComputeLocalized_AtHighTemperatureMatchesComputeOnTheAveragePooledEmbedding()
    {
        // dot(mean_i(location_i), tag) == mean_i(dot(location_i, tag)) by linearity, so at
        // temperature -> infinity the log-sum-exp pool over locations should converge to
        // exactly the same logits (and therefore loss) Compute gets from average-pooling
        // the spatial map first -- the property the whole localization loss is built on.
        float[] tagEmbedding = [1f, 0.5f];
        var spatialData = new float[]
        {
            // location (0,0)      (0,1)
            1f, 0f, /**/ 0f, 1f,
            // location (1,0)      (1,1)
            0.5f, 0.5f, /**/ -1f, 2f
        };
        var spatial = tensor(spatialData, [1, 2, 2, 2]);
        var tagEmbeddings = tensor(tagEmbedding, [1, 2]);
        var labels = tensor(new float[] { 1f }, [1, 1]);

        var pooledEmbedding = spatial.mean([2, 3]);

        var globalLoss = SigmoidContrastiveLoss.Compute(pooledEmbedding, tagEmbeddings, labels).item<float>();
        var localizedLoss = SigmoidContrastiveLoss.ComputeLocalized(spatial, tagEmbeddings, labels, temperature: 1000f).item<float>();

        Assert.Equal(globalLoss, localizedLoss, 3);
    }

    [Fact]
    public void ComputeLocalized_MaskedLocationCannotBePickedAsTheBestMatch()
    {
        // Single-channel spatial map so each of the 4 flattened (h*W+w) locations has an
        // unambiguous scalar logit: location index 1 aligns best with the tag but is
        // masked out (letterbox padding). The loss should reflect only the unmasked
        // locations, none of which align well -- i.e. it should score as a poor match
        // despite the raw heatmap's peak.
        float[] tagEmbedding = [1f];
        var spatialData = new float[] { -5f, 5f, -5f, -5f };
        var spatial = tensor(spatialData, [1, 1, 2, 2]);
        var tagEmbeddings = tensor(tagEmbedding, [1, 1]);
        var labels = tensor(new float[] { 1f }, [1, 1]);
        var mask = tensor(new float[] { 1f, 0f, 1f, 1f }, [1, 1, 2, 2]);

        var maskedLoss = SigmoidContrastiveLoss.ComputeLocalized(spatial, tagEmbeddings, labels, temperature: 0.1f, mask).item<float>();
        var unmaskedLoss = SigmoidContrastiveLoss.ComputeLocalized(spatial, tagEmbeddings, labels, temperature: 0.1f).item<float>();

        Assert.True(maskedLoss > unmaskedLoss);
    }

    [Fact]
    public void ComputeLocalized_WithAnAllValidMask_MatchesTheUnmaskedResult()
    {
        float[] tagEmbedding = [1f, 0f];
        var spatialData = new float[] { 1f, 0f, -1f, 0f, 0f, 1f, 0.5f, 0.5f };
        var spatial = tensor(spatialData, [1, 2, 2, 2]);
        var tagEmbeddings = tensor(tagEmbedding, [1, 2]);
        var labels = tensor(new float[] { 1f }, [1, 1]);
        var allValidMask = ones([1, 1, 2, 2]);

        var maskedLoss = SigmoidContrastiveLoss.ComputeLocalized(spatial, tagEmbeddings, labels, temperature: 0.5f, allValidMask).item<float>();
        var unmaskedLoss = SigmoidContrastiveLoss.ComputeLocalized(spatial, tagEmbeddings, labels, temperature: 0.5f).item<float>();

        Assert.Equal(unmaskedLoss, maskedLoss, 4);
    }

    [Fact]
    public void ComputeLocalized_AtLowTemperatureIsDrivenByTheBestMatchingLocationAlone()
    {
        // One location aligns closely with the tag, the rest are orthogonal/negative. A low
        // (max-like) temperature should score this as a much better match than a high
        // (mean-like) temperature would, since the mean is dragged down by the other locations.
        float[] tagEmbedding = [1f, 0f];
        var spatialData = new float[]
        {
            5f, 0f, /**/ -5f, 0f,
            0f, -5f, /**/ -5f, 5f
        };
        var spatial = tensor(spatialData, [1, 2, 2, 2]);
        var tagEmbeddings = tensor(tagEmbedding, [1, 2]);
        var labels = tensor(new float[] { 1f }, [1, 1]);

        var sharpLoss = SigmoidContrastiveLoss.ComputeLocalized(spatial, tagEmbeddings, labels, temperature: 0.1f).item<float>();
        var smoothLoss = SigmoidContrastiveLoss.ComputeLocalized(spatial, tagEmbeddings, labels, temperature: 1000f).item<float>();

        Assert.True(sharpLoss < smoothLoss);
    }

    // Fixed, arbitrary data shared by the chunked-vs-unchunked tests below: batch=2,
    // dim=2, a 2x2 spatial grid, vocabSize=5. 5 doesn't divide evenly by the chunkSize
    // (2) these tests use, so every one of them exercises a genuine partial trailing
    // chunk (tags [4,5)) -- the case most likely to reveal a chunk-weighting bug (see
    // ComputeLocalizedChunked's own doc comment on why chunks are combined by summing,
    // not by averaging per-chunk means).
    private const int ChunkedTestChunkSize = 2;
    private static readonly float[] ChunkedTestTagEmbeddingData = [1f, 0f, 0.5f, 0.5f, -1f, 1f, 0f, 1f, 0.3f, -0.7f];
    private static readonly float[] ChunkedTestSpatialData =
    [
        1f, 0f, 0.5f, -0.5f, 0f, 1f, -1f, 0.2f,
        0.2f, 0.3f, -0.4f, 0.6f, 1f, -1f, 0.5f, 0.1f
    ];
    private static readonly float[] ChunkedTestLabelData = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1];

    [Fact]
    public void ComputeLocalizedChunked_MatchesTheUnchunkedLossValue()
    {
        var spatial = tensor(ChunkedTestSpatialData, [2, 2, 2, 2]);
        // requires_grad_ on at least one input: ComputeLocalizedChunked always calls
        // backward() internally, which needs something in the graph to backprop into
        // (a real training call always has this, since the image/tag towers are
        // trainable) -- ComputeLocalized has no such requirement since it just returns
        // a Tensor for the caller to optionally backward() themselves.
        var tagEmbeddings = tensor(ChunkedTestTagEmbeddingData, [5, 2]).requires_grad_();
        var labels = tensor(ChunkedTestLabelData, [2, 5]);

        var expected = SigmoidContrastiveLoss.ComputeLocalized(spatial, tagEmbeddings, labels, temperature: 0.7f).item<float>();
        var chunked = SigmoidContrastiveLoss.ComputeLocalizedChunked(spatial, tagEmbeddings, labels, temperature: 0.7f, tagChunkSize: ChunkedTestChunkSize, gradientScale: 1f);

        Assert.Equal(expected, chunked, 4);
    }

    [Fact]
    public void ComputeLocalizedChunked_WithAMask_MatchesTheUnchunkedMaskedLossValue()
    {
        var spatial = tensor(ChunkedTestSpatialData, [2, 2, 2, 2]);
        var tagEmbeddings = tensor(ChunkedTestTagEmbeddingData, [5, 2]).requires_grad_();
        var labels = tensor(ChunkedTestLabelData, [2, 5]);
        var mask = tensor(new float[] { 1, 0, 1, 1, 1, 1, 0, 1 }, [2, 1, 2, 2]);

        var expected = SigmoidContrastiveLoss.ComputeLocalized(spatial, tagEmbeddings, labels, temperature: 0.7f, mask).item<float>();
        var chunked = SigmoidContrastiveLoss.ComputeLocalizedChunked(spatial, tagEmbeddings, labels, temperature: 0.7f, tagChunkSize: ChunkedTestChunkSize, gradientScale: 1f, mask);

        Assert.Equal(expected, chunked, 4);
    }

    /// <summary>
    /// The critical correctness proof for chunking: the same starting values, gradients
    /// computed two different ways (one external backward() call on the unchunked loss
    /// vs. ComputeLocalizedChunked's own internal per-chunk backward() calls), must land
    /// on the image tower's and tag tower's leaves identically. If chunking silently
    /// changed the math — not just reorganized it — this is where it would show up,
    /// since a loss-VALUE match alone (the tests above) doesn't guarantee the gradients
    /// backward() actually produced are also correct.
    /// </summary>
    [Fact]
    public void ComputeLocalizedChunked_ProducesTheSameGradientsAsTheUnchunkedVersion()
    {
        var labels = tensor(ChunkedTestLabelData, [2, 5]);

        var spatialUnchunked = tensor(ChunkedTestSpatialData, [2, 2, 2, 2]).requires_grad_();
        var tagEmbeddingsUnchunked = tensor(ChunkedTestTagEmbeddingData, [5, 2]).requires_grad_();
        SigmoidContrastiveLoss.ComputeLocalized(spatialUnchunked, tagEmbeddingsUnchunked, labels, temperature: 0.7f).backward();

        var spatialChunked = tensor(ChunkedTestSpatialData, [2, 2, 2, 2]).requires_grad_();
        var tagEmbeddingsChunked = tensor(ChunkedTestTagEmbeddingData, [5, 2]).requires_grad_();
        SigmoidContrastiveLoss.ComputeLocalizedChunked(spatialChunked, tagEmbeddingsChunked, labels, temperature: 0.7f, tagChunkSize: ChunkedTestChunkSize, gradientScale: 1f);

        AssertAllClose(spatialUnchunked.grad!, spatialChunked.grad!);
        AssertAllClose(tagEmbeddingsUnchunked.grad!, tagEmbeddingsChunked.grad!);
    }

    [Fact]
    public void ComputeLocalizedChunked_GradientScaleLinearlyScalesTheGradient()
    {
        var labels = tensor(ChunkedTestLabelData, [2, 5]);

        var spatialFull = tensor(ChunkedTestSpatialData, [2, 2, 2, 2]).requires_grad_();
        var tagEmbeddingsFull = tensor(ChunkedTestTagEmbeddingData, [5, 2]).requires_grad_();
        SigmoidContrastiveLoss.ComputeLocalizedChunked(spatialFull, tagEmbeddingsFull, labels, temperature: 0.7f, tagChunkSize: ChunkedTestChunkSize, gradientScale: 1f);

        var spatialQuarter = tensor(ChunkedTestSpatialData, [2, 2, 2, 2]).requires_grad_();
        var tagEmbeddingsQuarter = tensor(ChunkedTestTagEmbeddingData, [5, 2]).requires_grad_();
        SigmoidContrastiveLoss.ComputeLocalizedChunked(spatialQuarter, tagEmbeddingsQuarter, labels, temperature: 0.7f, tagChunkSize: ChunkedTestChunkSize, gradientScale: 0.25f);

        AssertAllClose(spatialFull.grad!.mul(0.25f), spatialQuarter.grad!);
        AssertAllClose(tagEmbeddingsFull.grad!.mul(0.25f), tagEmbeddingsQuarter.grad!);
    }

    private static void AssertAllClose(Tensor expected, Tensor actual)
    {
        var expectedValues = expected.data<float>().ToArray();
        var actualValues = actual.data<float>().ToArray();
        Assert.Equal(expectedValues.Length, actualValues.Length);
        for (var i = 0; i < expectedValues.Length; i++)
            Assert.Equal(expectedValues[i], actualValues[i], 4);
    }
}

public class SigmoidContrastiveLossWeightedTests
{
    private static PairTargets Targets(float[] labels, float[] weights, float[] focal, int images, int tags) =>
        new(
            tensor(labels, [images, tags]), tensor(weights, [images, tags]), tensor(focal, [images, tags]),
            labels.Zip(weights).Where(x => x.First > 0).Sum(x => x.Second));

    [Fact]
    public void ComputeWeighted_WithoutFocalOrWeights_IsTheSummedBceOverThePositiveCount()
    {
        var image = tensor(new[] { 1f, 0f }, [1, 2]);
        var tags = tensor(new[] { 1f, 0f, 0f, 1f, -1f, 0f }, [3, 2]);
        using var targets = Targets([1f, -1f, -1f], [1f, 1f, 1f], [0f, 0f, 0f], 1, 3);
        var labels = tensor(new[] { 1f, -1f, -1f }, [1, 3]);

        var weighted = SigmoidContrastiveLoss.ComputeWeighted(image, tags, targets, null, negativeGamma: 4f, negativeClip: 0.05f).item<float>();
        var plain = SigmoidContrastiveLoss.Compute(image, tags, labels).item<float>() * 3; // mean over 3 pairs -> sum

        Assert.Equal(plain, weighted, 4);
    }

    [Fact]
    public void ComputeWeighted_ZeroWeightPairsContributeNothing()
    {
        var image = tensor(new[] { 1f, 0f }, [1, 2]);
        var tags = tensor(new[] { 1f, 0f, 5f, 0f }, [2, 2]);
        using var withPair = Targets([1f, -1f], [1f, 1f], [0f, 0f], 1, 2);
        using var withoutPair = Targets([1f, -1f], [1f, 0f], [0f, 0f], 1, 2);
        using var positiveOnly = Targets([1f], [1f], [0f], 1, 1);

        var dropped = SigmoidContrastiveLoss.ComputeWeighted(image, tags, withoutPair, null, 4f, 0.05f).item<float>();
        var kept = SigmoidContrastiveLoss.ComputeWeighted(image, tags, withPair, null, 4f, 0.05f).item<float>();
        var reference = SigmoidContrastiveLoss.ComputeWeighted(image, tags.narrow(0, 0, 1), positiveOnly, null, 4f, 0.05f).item<float>();

        Assert.Equal(reference, dropped, 5);
        Assert.True(kept > dropped);
    }

    [Fact]
    public void ComputeWeighted_UnverifiedNegativeBelowTheClipCostsNothing()
    {
        var image = tensor(new[] { 1f }, [1, 1]);
        var tags = tensor(new[] { -10f }, [1, 1]); // sigmoid(-10) ~ 4.5e-5, far below a 0.05 clip
        using var unverified = Targets([-1f], [1f], [1f], 1, 1);
        using var certain = Targets([-1f], [1f], [0f], 1, 1);

        Assert.Equal(0f, SigmoidContrastiveLoss.ComputeWeighted(image, tags, unverified, null, 4f, 0.05f).item<float>());
        Assert.True(SigmoidContrastiveLoss.ComputeWeighted(image, tags, certain, null, 4f, 0.05f).item<float>() > 0f);
    }

    [Fact]
    public void ComputeWeighted_DampsAnUnverifiedNegativeRelativeToACertainOne()
    {
        var image = tensor(new[] { 1f }, [1, 1]);
        var tags = tensor(new[] { 0f }, [1, 1]); // p = 0.5
        using var unverified = Targets([-1f], [1f], [1f], 1, 1);
        using var certain = Targets([-1f], [1f], [0f], 1, 1);

        var damped = SigmoidContrastiveLoss.ComputeWeighted(image, tags, unverified, null, 4f, 0.05f).item<float>();
        var full = SigmoidContrastiveLoss.ComputeWeighted(image, tags, certain, null, 4f, 0.05f).item<float>();

        Assert.True(damped < full / 10);
    }

    [Fact]
    public void ComputeLocalizedChunked_WithTargets_MatchesAcrossChunkSizes()
    {
        var generator = new Random(5);
        var spatial = tensor(Enumerable.Range(0, 2 * 3 * 2 * 2).Select(_ => (float)generator.NextDouble() - 0.5f).ToArray(), [2, 3, 2, 2], requires_grad: true);
        var tags = tensor(Enumerable.Range(0, 5 * 3).Select(_ => (float)generator.NextDouble() - 0.5f).ToArray(), [5, 3], requires_grad: true);
        var labels = new[] { 1f, -1f, -1f, 1f, -1f, -1f, 1f, -1f, -1f, -1f };
        using var targets = Targets(labels, [1f, 1f, 0f, 1f, 2f, 1f, 1f, 1f, 0f, 1f], [0f, 1f, 1f, 0f, 0f, 1f, 0f, 1f, 1f, 1f], 2, 5);
        var plainLabels = tensor(labels, [2, 5]);

        var whole = SigmoidContrastiveLoss.ComputeLocalizedChunked(spatial, tags, plainLabels, 0.5f, 5, 0f, targets: targets, negativeGamma: 2f, negativeClip: 0f);
        var chunked = SigmoidContrastiveLoss.ComputeLocalizedChunked(spatial, tags, plainLabels, 0.5f, 2, 0f, targets: targets, negativeGamma: 2f, negativeClip: 0f);

        Assert.Equal(whole, chunked, 5);
    }
}
