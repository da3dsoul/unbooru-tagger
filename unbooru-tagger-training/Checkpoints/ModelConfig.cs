namespace UnbooruTagger.Training.Checkpoints;

/// <summary>The <see cref="Model.ImageTower"/> shape needed to reconstruct an identical module before loading trained weights into it.</summary>
public sealed record ModelConfig(int EmbeddingDim, int InputSize, int StemChannels, int[] StageChannels, int[] BlocksPerStage)
{
    public static ModelConfig Default(int embeddingDim, int inputSize) =>
        new(embeddingDim, inputSize, 64, [64, 128, 256], [2, 2, 2]);
}
