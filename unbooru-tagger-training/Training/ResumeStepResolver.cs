using UnbooruTagger.Training.Checkpoints;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Decides how many steps of the epoch being resumed into were already done — pulled
/// out of <see cref="Commands.TrainCommandHandler"/>'s loop as its own pure, stateless
/// function so this specific decision is independently testable.
///
/// The bug this exists to prevent: <see cref="TrainingProgress.CompletedStepsInEpoch"/>
/// is only meaningful relative to the step layout (<c>trainingIndices.Count /
/// batchSize</c>) that was in effect when it was saved. If <c>--batch-size</c> (or the
/// dataset size) changes across a resume, that layout changes too, and the old count is
/// no longer comparable to the new one — it can land anywhere relative to the new,
/// different <paramref name="stepsPerEpoch"/>, including at or past it. Naively
/// clamping it (<c>Math.Min(old, new)</c>) used to mean landing exactly at the new
/// <paramref name="stepsPerEpoch"/> silently ran that epoch's training loop zero times
/// (its `for` loop's start already equaled its end) — no error, no warning, just an
/// epoch that consumed a validation pass and a "completed" checkpoint without ever
/// training under the new step layout.
/// </summary>
public static class ResumeStepResolver
{
    public readonly record struct Result(int StepsInEpoch, bool StepLayoutChanged);

    public static Result Resolve(bool resumingOptimizerState, TrainingProgress resumedProgress, int stepsPerEpoch)
    {
        if (!resumingOptimizerState)
            return new Result(0, StepLayoutChanged: false);

        // An older checkpoint missing StepsPerEpoch deserializes it as 0 (TrainingProgress's
        // own default), which never equals a real stepsPerEpoch (always >= 1) -- so it's
        // treated the same as a genuine mismatch: safer to restart progress within the
        // epoch than to risk clamping into the silent-skip state described above.
        if (resumedProgress.StepsPerEpoch == stepsPerEpoch)
            return new Result(resumedProgress.CompletedStepsInEpoch, StepLayoutChanged: false);

        return new Result(0, StepLayoutChanged: true);
    }
}
