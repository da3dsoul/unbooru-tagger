using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

// SigmoidContrastiveLoss.HillNegatives is static: these must not run alongside any other loss test.
[CollectionDefinition(nameof(HillNegativesTests), DisableParallelization = true)]
public class HillNegativesCollection;

[Collection(nameof(HillNegativesTests))]
public class HillNegativesTests
{
    private static (float Loss, float Gradient) Negative(float logit, bool hill)
    {
        var previous = SigmoidContrastiveLoss.HillNegatives;
        SigmoidContrastiveLoss.HillNegatives = hill;
        try
        {
            using var logits = tensor(new[] { logit }, [1, 1]).requires_grad_(true);
            using var targets = new PairTargets(tensor(new[] { -1f }, [1, 1]), ones(1, 1), ones(1, 1), 1f);
            var loss = SigmoidContrastiveLoss.ComputeWeightedScores(logits, targets, null, negativeGamma: 0, negativeClip: 0);
            loss.backward();
            return (loss.item<float>(), logits.grad!.item<float>());
        }
        finally
        {
            SigmoidContrastiveLoss.HillNegatives = previous;
        }
    }

    [Fact]
    public void Hill_IsOnePointFiveMinusP_TimesPSquared()
    {
        Assert.Equal((1.5f - 0.5f) * 0.25f, Negative(0f, hill: true).Loss, 5);
        Assert.Equal(MathF.Log(2), Negative(0f, hill: false).Loss, 5); // the plain BCE it replaces, for scale
    }

    [Fact]
    public void Hill_StopsPushingOnAConfidentYes_WhereBceKeepsPushingHarder()
    {
        // p = 0.99: likely an unlabelled object. BCE's gradient is ~p; Hill's falls towards 0.
        var (_, bce) = Negative(4.6f, hill: false);
        var (_, hill) = Negative(4.6f, hill: true);
        Assert.True(bce > 0.98f, $"BCE gradient {bce}");
        Assert.True(hill < 0.05f, $"Hill gradient {hill}");
        // It peaks around p = 0.5 instead.
        Assert.True(Negative(0f, hill: true).Gradient > 5 * hill);
    }
}
