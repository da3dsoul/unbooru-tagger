using System.Text.Json;
using TorchSharp.Modules;

namespace UnbooruTagger.Training.Checkpoints;

/// <summary>
/// How many epochs a run has completed, plus <see cref="Training.EarlyStopping"/>'s
/// evaluation history at that point. Neither lives in the model weights <see cref="Checkpoint"/>
/// covers, so without this a resumed run continues the model from where it left off
/// but silently restarts the epoch count from zero and gives EarlyStopping a fresh
/// patience window instead of the one it was partway through.
/// </summary>
public sealed record TrainingProgress(int CompletedEpochs, double EarlyStoppingBestLoss, int EarlyStoppingEvaluationsSinceImprovement)
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

    public static bool Exists(string directory) =>
        File.Exists(Path.Combine(directory, ProgressFileName)) &&
        File.Exists(Path.Combine(directory, OptimizerFileName));

    public static void Save(string directory, TrainingProgress progress, OptimizerHelper optimizer)
    {
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, ProgressFileName), JsonSerializer.Serialize(progress, new JsonSerializerOptions { WriteIndented = true }));
        optimizer.save_state_dict(Path.Combine(directory, OptimizerFileName));
    }

    public static TrainingProgress LoadProgress(string directory) =>
        JsonSerializer.Deserialize<TrainingProgress>(File.ReadAllText(Path.Combine(directory, ProgressFileName)))
        ?? throw new InvalidDataException($"'{directory}' does not contain valid training progress.");

    /// <summary>
    /// Loads Adam's per-parameter state into <paramref name="optimizer"/> in place.
    /// The optimizer must already be constructed against the same parameters (same
    /// shapes, same order) that produced the saved state — <see cref="UnbooruTagger.Training.Commands.TrainCommandHandler"/>
    /// only calls this once the resumed image/tag towers are already built.
    /// </summary>
    public static void LoadOptimizerState(string directory, OptimizerHelper optimizer) =>
        optimizer.load_state_dict(Path.Combine(directory, OptimizerFileName));
}
