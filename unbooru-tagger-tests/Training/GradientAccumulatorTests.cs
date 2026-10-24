using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class GradientAccumulatorTests
{
    [Fact]
    public void NoAccumulation_ZeroesAndStepsEveryMicroStep()
    {
        for (var step = 0; step < 5; step++)
        {
            Assert.True(GradientAccumulator.ShouldZeroGrad(step, accumulationSteps: 1));
            Assert.True(GradientAccumulator.ShouldStep(step, accumulationSteps: 1, stepsPerEpoch: 5));
        }
    }

    [Fact]
    public void ShouldZeroGrad_OnlyAtTheStartOfEachCycle()
    {
        Assert.True(GradientAccumulator.ShouldZeroGrad(0, accumulationSteps: 4));
        Assert.False(GradientAccumulator.ShouldZeroGrad(1, accumulationSteps: 4));
        Assert.False(GradientAccumulator.ShouldZeroGrad(2, accumulationSteps: 4));
        Assert.False(GradientAccumulator.ShouldZeroGrad(3, accumulationSteps: 4));
        Assert.True(GradientAccumulator.ShouldZeroGrad(4, accumulationSteps: 4));
        Assert.True(GradientAccumulator.ShouldZeroGrad(8, accumulationSteps: 4));
    }

    [Fact]
    public void ShouldStep_OnlyAtTheEndOfEachCompleteCycle()
    {
        Assert.False(GradientAccumulator.ShouldStep(0, accumulationSteps: 4, stepsPerEpoch: 100));
        Assert.False(GradientAccumulator.ShouldStep(1, accumulationSteps: 4, stepsPerEpoch: 100));
        Assert.False(GradientAccumulator.ShouldStep(2, accumulationSteps: 4, stepsPerEpoch: 100));
        Assert.True(GradientAccumulator.ShouldStep(3, accumulationSteps: 4, stepsPerEpoch: 100));
        Assert.False(GradientAccumulator.ShouldStep(4, accumulationSteps: 4, stepsPerEpoch: 100));
        Assert.True(GradientAccumulator.ShouldStep(7, accumulationSteps: 4, stepsPerEpoch: 100));
    }

    /// <summary>
    /// stepsPerEpoch = 10, accumulationSteps = 4: full cycles complete at steps 3 and 7,
    /// leaving a trailing partial cycle of steps 8-9 (only 2 of 4 micro-batches). That
    /// partial cycle must still flush at the epoch's last step instead of silently
    /// discarding those two micro-batches' accumulated gradient.
    /// </summary>
    [Fact]
    public void ShouldStep_FlushesATrailingPartialCycleAtTheEpochsLastStep()
    {
        const int accumulationSteps = 4;
        const int stepsPerEpoch = 10;

        Assert.True(GradientAccumulator.ShouldStep(3, accumulationSteps, stepsPerEpoch));
        Assert.True(GradientAccumulator.ShouldStep(7, accumulationSteps, stepsPerEpoch));
        Assert.False(GradientAccumulator.ShouldStep(8, accumulationSteps, stepsPerEpoch));
        Assert.True(GradientAccumulator.ShouldStep(9, accumulationSteps, stepsPerEpoch));
    }
}
