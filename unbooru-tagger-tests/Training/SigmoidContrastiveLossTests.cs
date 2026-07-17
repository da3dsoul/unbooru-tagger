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
}
