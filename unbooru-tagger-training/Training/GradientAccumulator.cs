namespace UnbooruTagger.Training.Training;

/// <summary>
/// Decides, for micro-batch index <c>step</c> within an epoch's <c>stepsPerEpoch</c>
/// total micro-batches, whether gradients should be zeroed before this micro-batch's
/// backward pass (the start of a new accumulation cycle) and whether the optimizer
/// should step after it (the end of a cycle — including a flushed partial cycle at the
/// epoch's last micro-batch, so a trailing few micro-batches that don't fill a whole
/// cycle still contribute instead of being silently dropped). Kept as its own pure,
/// stateless scheduler — separate from <see cref="Commands.TrainCommandHandler"/>'s loop
/// — so this modulo/edge-case reasoning is independently testable instead of only
/// exercised by running a real training loop.
///
/// Exists so a machine that can only fit a small micro-batch in memory (e.g.
/// <c>--batch-size 8</c>) can still train against a larger effective batch
/// (<c>--gradient-accumulation-steps N</c> gives an effective batch of
/// <c>batchSize * N</c>) — a bigger effective batch means more positive rows per
/// optimizer step in the sigmoid-contrastive label matrix, which matters more the
/// smaller <c>--batch-size</c> is forced to be (see <see cref="SigmoidContrastiveLoss"/>'s
/// own doc comment on why that ratio matters).
/// </summary>
public static class GradientAccumulator
{
    public static bool ShouldZeroGrad(int step, int accumulationSteps) => step % accumulationSteps == 0;

    public static bool ShouldStep(int step, int accumulationSteps, int stepsPerEpoch) =>
        (step + 1) % accumulationSteps == 0 || step == stepsPerEpoch - 1;
}
