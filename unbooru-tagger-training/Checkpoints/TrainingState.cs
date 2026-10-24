using System.Text.Json;
using System.Text.Json.Serialization;
using TorchSharp.Modules;

namespace UnbooruTagger.Training.Checkpoints;

/// <summary>
/// How many epochs a run has completed, plus <see cref="Training.EarlyStopping"/>'s
/// evaluation history at that point. Neither lives in the model weights <see cref="Checkpoint"/>
/// covers, so without this a resumed run continues the model from where it left off
/// but silently restarts the epoch count from zero and gives EarlyStopping a fresh
/// patience window instead of the one it was partway through.
///
/// <paramref name="CompletedStepsInEpoch"/> and <paramref name="ValidationSplitSeed"/>
/// exist for mid-epoch resume: an epoch can take multiple days, so <see cref="Commands.TrainCommandHandler"/>
/// also checkpoints periodically *within* an epoch, not just at epoch boundaries.
/// <paramref name="CompletedStepsInEpoch"/> is how many steps of the *current, not-yet-
/// finished* epoch were already done (0 once that epoch completes and
/// <paramref name="CompletedEpochs"/> advances instead). <paramref name="ValidationSplitSeed"/>
/// is persisted so a resumed run reconstructs the exact same train/validation split
/// instead of re-shuffling — otherwise a mid-epoch resume could train on former
/// validation images. All three default to 0 so an older checkpoint written before
/// they existed still deserializes and resumes (just at an epoch boundary, with a
/// re-derived split, same as before).
///
/// <paramref name="StepsPerEpoch"/> is how many micro-batches made up one epoch when
/// this was saved — i.e. <c>trainingIndices.Count / batchSize</c> at save time. It
/// exists purely so a resume can tell whether <paramref name="CompletedStepsInEpoch"/>
/// is still comparable to *this* run's own step layout: if <c>--batch-size</c> (or the
/// dataset size) changed across the resume, the old count can land anywhere relative
/// to the new, different <c>stepsPerEpoch</c> — including past it, which used to mean
/// the resumed epoch's training loop silently ran zero iterations (clamped straight to
/// "epoch already done," see <see cref="Commands.TrainCommandHandler"/>). Defaults to 0
/// like the others, which deliberately does NOT match any real freshly-computed
/// <c>stepsPerEpoch</c> (always >= 1) — so an older checkpoint missing this field is
/// treated the same as a genuine mismatch: safely restart progress within the epoch
/// from 0 rather than risk clamping into that silent-skip state.
///
/// <c>EarlyStoppingMetric</c> names what <c>EarlyStoppingBestLoss</c> measures; null for a
/// checkpoint written before it existed (when it was always mean validation loss). A resume
/// only restores early-stopping history recorded against the metric the current build uses.
/// </summary>
public sealed record TrainingProgress(
    int CompletedEpochs,
    double EarlyStoppingBestLoss,
    int EarlyStoppingEvaluationsSinceImprovement,
    int CompletedStepsInEpoch = 0,
    int ValidationSplitSeed = 0,
    int StepsPerEpoch = 0,
    string? EarlyStoppingMetric = null)
{
    public static TrainingProgress Initial => new(0, double.PositiveInfinity, 0);
}

/// <summary>
/// Persists <see cref="TrainingProgress"/> alongside the optimizer's own state (Adam's
/// per-parameter momentum/variance) — separate from <see cref="Checkpoint"/> since only
/// the full/periodic `train` pass has a notion of epochs or early stopping to resume;
/// `add-tag`'s single-row fine-tune doesn't use this.
/// </summary>
public static class TrainingState
{
    private const string ProgressFileName = "training_progress.json";
    private const string OptimizerFileName = "optimizer.dat";
    private const string LogitScaleBiasOptimizerFileName = "logit_scale_bias_optimizer.dat";

    // EarlyStoppingBestLoss starts at double.PositiveInfinity (see TrainingProgress.Initial),
    // which System.Text.Json rejects as a number literal unless explicitly allowed.
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    public static bool Exists(string directory) =>
        File.Exists(Path.Combine(directory, ProgressFileName)) &&
        File.Exists(Path.Combine(directory, OptimizerFileName));

    public static void Save(string directory, TrainingProgress progress, OptimizerHelper optimizer)
    {
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, ProgressFileName), JsonSerializer.Serialize(progress, JsonOptions));
        optimizer.save_state_dict(Path.Combine(directory, OptimizerFileName));
    }

    /// <summary>Whether <paramref name="directory"/> has training_progress.json, with or without an optimizer (an offloaded-evaluation snapshot has none).</summary>
    public static bool HasProgress(string directory) => File.Exists(Path.Combine(directory, ProgressFileName));

    /// <summary>Writes training_progress.json alone: enough for <c>evaluate</c> to reconstruct the run's validation split.</summary>
    public static void SaveProgress(string directory, TrainingProgress progress)
    {
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, ProgressFileName), JsonSerializer.Serialize(progress, JsonOptions));
    }

    public static TrainingProgress LoadProgress(string directory) =>
        JsonSerializer.Deserialize<TrainingProgress>(File.ReadAllText(Path.Combine(directory, ProgressFileName)), JsonOptions)
        ?? throw new InvalidDataException($"'{directory}' does not contain valid training progress.");

    /// <summary>
    /// Loads Adam's per-parameter state into <paramref name="optimizer"/> in place.
    /// The optimizer must already be constructed against the same parameters (same
    /// shapes, same order) that produced the saved state — <see cref="UnbooruTagger.Training.Commands.TrainCommandHandler"/>
    /// only calls this once the resumed image/tag towers are already built.
    /// </summary>
    public static void LoadOptimizerState(string directory, OptimizerHelper optimizer) =>
        optimizer.load_state_dict(Path.Combine(directory, OptimizerFileName));

    /// <summary>Whether <paramref name="directory"/>'s checkpoint has a saved optimizer for <c>LogitScaleBias</c> — false for one written before that module existed (see <see cref="Checkpoint.HasLogitScaleBias"/>'s doc comment for the same fallback reasoning).</summary>
    public static bool HasLogitScaleBiasOptimizerState(string directory) => File.Exists(Path.Combine(directory, LogitScaleBiasOptimizerFileName));

    /// <summary>
    /// <c>LogitScaleBias</c>'s optimizer is saved/loaded separately from the main
    /// image/tag-tower optimizer above — deliberately, not incidentally: folding its two
    /// scalar parameters into the same <c>optim.Adam(...)</c> call would change that
    /// optimizer's parameter ordering, and <see cref="LoadOptimizerState"/> restores Adam
    /// state positionally, so that would silently misalign — or outright fail to load —
    /// every bit of momentum already accumulated for the image tower and tag embeddings
    /// across prior epochs. Keeping it a fully independent optimizer means an old
    /// checkpoint's <c>optimizer.dat</c> (and everything it represents) loads completely
    /// unaffected by <c>LogitScaleBias</c> not having existed yet.
    /// </summary>
    public static void SaveLogitScaleBiasOptimizerState(string directory, OptimizerHelper optimizer)
    {
        Directory.CreateDirectory(directory);
        optimizer.save_state_dict(Path.Combine(directory, LogitScaleBiasOptimizerFileName));
    }

    public static void LoadLogitScaleBiasOptimizerState(string directory, OptimizerHelper optimizer) =>
        optimizer.load_state_dict(Path.Combine(directory, LogitScaleBiasOptimizerFileName));
}
