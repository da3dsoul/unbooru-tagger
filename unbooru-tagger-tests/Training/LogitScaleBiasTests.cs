using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class LogitScaleBiasTests
{
    [Fact]
    public void DefaultConstruction_ResolvesToTheDocumentedInitialScaleAndBias()
    {
        var scaleBias = new LogitScaleBias();

        Assert.Equal(10f, scaleBias.Scale, 4);
        Assert.Equal(-10f, scaleBias.Bias, 4);
    }

    [Fact]
    public void Forward_AppliesScaleThenBias()
    {
        var scaleBias = new LogitScaleBias(initialScale: 2f, initialBias: -1f);
        var logits = tensor(new float[] { 0f, 1f, 2f });

        var result = scaleBias.forward(logits);

        Assert.Equal(new float[] { -1f, 1f, 3f }, result.data<float>().ToArray());
    }

    [Fact]
    public void ToLogitScaling_MatchesTheResolvedScaleAndBias()
    {
        var scaleBias = new LogitScaleBias(initialScale: 5f, initialBias: -3f);

        var scaling = scaleBias.ToLogitScaling();

        Assert.Equal(5f, scaling.Scale, 4);
        Assert.Equal(-3f, scaling.Bias, 4);
    }

    /// <summary>
    /// Scale is stored in log-space specifically so gradient descent can't push it
    /// through zero (see the class's own doc comment) — this exercises that a real
    /// optimizer step still leaves it strictly positive even when the loss pushes hard
    /// in the direction that would otherwise flip its sign.
    /// </summary>
    [Fact]
    public void OptimizerStep_CannotPushScaleNonPositive()
    {
        // A large-but-not-underflowing gradient step: d(loss)/d(logScale) = 10 *
        // exp(0) = 10, so one lr=1 step drives log-scale from 0 down to -10 -- a
        // dramatic drop (Scale: 1 -> ~4.5e-5) without exp() underflowing all the way to
        // a literal 0.0f, which would trivially (and uninterestingly) satisfy "> 0".
        var scaleBias = new LogitScaleBias(initialScale: 1f, initialBias: 0f);
        var optimizer = optim.SGD(scaleBias.parameters().ToArray(), learningRate: 1.0);
        var logits = tensor(new float[] { 10f });

        var loss = scaleBias.forward(logits).sum();
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        Assert.True(scaleBias.Scale > 0f);
        Assert.True(scaleBias.Scale < 1f);
    }
}
