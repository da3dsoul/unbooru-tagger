using UnbooruTagger.Training.Checkpoints;
using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class ResumeStepResolverTests
{
    [Fact]
    public void Resolve_ReturnsZeroForAFreshRun()
    {
        var progress = new TrainingProgress(2, 0.5, 0, CompletedStepsInEpoch: 500, StepsPerEpoch: 1000);

        var result = ResumeStepResolver.Resolve(resumingOptimizerState: false, progress, stepsPerEpoch: 1000);

        Assert.Equal(0, result.StepsInEpoch);
        Assert.False(result.StepLayoutChanged);
    }

    [Fact]
    public void Resolve_TrustsTheSavedStepCountWhenTheStepLayoutMatches()
    {
        var progress = new TrainingProgress(2, 0.5, 0, CompletedStepsInEpoch: 500, StepsPerEpoch: 1000);

        var result = ResumeStepResolver.Resolve(resumingOptimizerState: true, progress, stepsPerEpoch: 1000);

        Assert.Equal(500, result.StepsInEpoch);
        Assert.False(result.StepLayoutChanged);
    }

    /// <summary>
    /// The exact scenario that used to silently run the resumed epoch's training loop
    /// zero times: a batch-size change roughly halves stepsPerEpoch, so a
    /// CompletedStepsInEpoch recorded under the old (larger) step count now lands past
    /// the new one. Math.Min(old, new) used to clamp this straight to "epoch already
    /// done." This must instead restart the epoch's progress from 0 and say so.
    /// </summary>
    [Fact]
    public void Resolve_RestartsFromZeroWhenTheStepLayoutChanged()
    {
        // Recorded late in a 1000-step epoch (--batch-size 8); resuming with
        // --batch-size 16 halves stepsPerEpoch to 500 -- the old count (900) would
        // clamp to exactly 500, the new stepsPerEpoch, running zero training iterations.
        var progress = new TrainingProgress(2, 0.5, 0, CompletedStepsInEpoch: 900, StepsPerEpoch: 1000);

        var result = ResumeStepResolver.Resolve(resumingOptimizerState: true, progress, stepsPerEpoch: 500);

        Assert.Equal(0, result.StepsInEpoch);
        Assert.True(result.StepLayoutChanged);
    }

    [Fact]
    public void Resolve_RestartsFromZeroForACheckpointThatPredatesTrackingStepsPerEpoch()
    {
        // StepsPerEpoch defaults to 0 (TrainingProgress's own default for a checkpoint
        // written before this field existed), which never equals a real stepsPerEpoch.
        var progress = new TrainingProgress(2, 0.5, 0, CompletedStepsInEpoch: 500);

        var result = ResumeStepResolver.Resolve(resumingOptimizerState: true, progress, stepsPerEpoch: 1000);

        Assert.Equal(0, result.StepsInEpoch);
        Assert.True(result.StepLayoutChanged);
    }
}
