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
}
