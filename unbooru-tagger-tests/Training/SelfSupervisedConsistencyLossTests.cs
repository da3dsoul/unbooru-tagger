using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class SelfSupervisedConsistencyLossTests
{
    [Fact]
    public void Compute_IsLowestWhenPredictionsPerfectlyMatchTheOtherViewsTarget()
    {
        var pooledA = tensor(new float[] { 1f, 0f }, [1, 2]);
        var pooledB = tensor(new float[] { 0f, 1f }, [1, 2]);

        // A perfect predictor maps each view's pooled embedding exactly onto the other's.
        var perfectPredictionA = pooledB.clone();
        var perfectPredictionB = pooledA.clone();
        var perfectLoss = SelfSupervisedConsistencyLoss.Compute(pooledA, perfectPredictionA, pooledB, perfectPredictionB).item<float>();

        // Anti-aligned with what each prediction is actually scored against: predictionA is
        // compared to targetB (pooledB), predictionB to targetA (pooledA) — see Compute's docs.
        var wrongPredictionA = pooledB.neg();
        var wrongPredictionB = pooledA.neg();
        var worstLoss = SelfSupervisedConsistencyLoss.Compute(pooledA, wrongPredictionA, pooledB, wrongPredictionB).item<float>();

        Assert.True(perfectLoss < worstLoss);
        Assert.Equal(-1f, perfectLoss, 3);
        Assert.Equal(1f, worstLoss, 3);
    }

    [Fact]
    public void Compute_IsZeroForOrthogonalPredictions()
    {
        var pooledA = tensor(new float[] { 1f, 0f }, [1, 2]);
        var pooledB = tensor(new float[] { 1f, 0f }, [1, 2]);
        var orthogonalPrediction = tensor(new float[] { 0f, 1f }, [1, 2]);

        var loss = SelfSupervisedConsistencyLoss.Compute(pooledA, orthogonalPrediction, pooledB, orthogonalPrediction).item<float>();

        Assert.Equal(0f, loss, 3);
    }
}
