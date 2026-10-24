namespace UnbooruTagger.Training.Checkpoints;

/// <summary>
/// The <see cref="Model.ImageTower"/> shape needed to reconstruct an identical module before
/// loading trained weights into it. <paramref name="FusedStages"/> defaults to 0 so a config
/// written before multi-scale fusion existed still loads as the original layout.
/// </summary>
public sealed record ModelConfig(int EmbeddingDim, int InputSize, int StemChannels, int[] StageChannels, int[] BlocksPerStage, int FusedStages = 0, bool LocalizationFromFusedOnly = false)
{
    public static ModelConfig Default(int embeddingDim, int inputSize) =>
        new(embeddingDim, inputSize, 64, [64, 128, 256], [2, 2, 2]);

    /// <summary>The config that reconstructs <paramref name="tower"/> exactly — what a checkpoint must save (not <see cref="Default"/>, which is only right for the original layout).</summary>
    public static ModelConfig From(Model.ImageTower tower, int inputSize) =>
        new(tower.EmbeddingDim, inputSize, tower.StemChannels, tower.StageChannels, tower.BlocksPerStage, tower.FusedStages, tower.LocalizationFromFusedOnly);
}
