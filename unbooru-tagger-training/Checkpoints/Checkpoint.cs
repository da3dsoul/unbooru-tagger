using System.Text.Json;
using UnbooruTagger.Core.Embedding;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Checkpoints;

/// <summary>
/// Everything needed to resume training or hand off to <c>export-onnx</c>: the image
/// tower weights (TorchSharp's own format, not ONNX — see
/// <see cref="Export.ImageTowerOnnxExporter"/> for that bridge), the config needed to
/// reconstruct the same module shape before loading weights into it, the tag
/// vocabulary, and the tag embedding table.
/// </summary>
public static class Checkpoint
{
    private const string ImageTowerFileName = "image_tower.dat";
    private const string ConfigFileName = "model_config.json";
    private const string VocabularyFileName = "tag_vocabulary.json";
    private const string EmbeddingsFileName = "tag_embeddings.bin";
    private const string LogitScaleBiasFileName = "logit_scale_bias.dat";

    /// <summary>Whether <paramref name="directory"/> holds a complete checkpoint <see cref="Load"/> can read back.</summary>
    public static bool Exists(string directory) =>
        File.Exists(Path.Combine(directory, ConfigFileName)) &&
        File.Exists(Path.Combine(directory, ImageTowerFileName)) &&
        File.Exists(Path.Combine(directory, VocabularyFileName)) &&
        File.Exists(Path.Combine(directory, EmbeddingsFileName));

    /// <summary>Whether <paramref name="directory"/>'s checkpoint was written by a build new enough to have a saved <see cref="LogitScaleBias"/> — see <see cref="Load"/>.</summary>
    public static bool HasLogitScaleBias(string directory) => File.Exists(Path.Combine(directory, LogitScaleBiasFileName));

    public static void Save(string directory, ImageTower imageTower, ModelConfig config, TagVocabulary vocabulary, TagEmbeddingStore embeddings, LogitScaleBias logitScaleBias)
    {
        Directory.CreateDirectory(directory);
        imageTower.save(Path.Combine(directory, ImageTowerFileName));
        File.WriteAllText(Path.Combine(directory, ConfigFileName), JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true }));
        vocabulary.Save(Path.Combine(directory, VocabularyFileName));
        embeddings.Save(Path.Combine(directory, EmbeddingsFileName));
        logitScaleBias.save(Path.Combine(directory, LogitScaleBiasFileName));
    }

    /// <summary>
    /// <see cref="LogitScaleBias"/> is always returned with a freshly constructed (default-init)
    /// module if <paramref name="directory"/> predates it (<see cref="HasLogitScaleBias"/> is
    /// false) — the checkpoint still resumes, just with those two scalars starting over
    /// instead of continuing from a previously-learned value, the same graceful-fallback
    /// pattern <see cref="Checkpoints.TrainingState"/> uses for an old checkpoint missing
    /// optimizer/progress state.
    /// </summary>
    public static (ImageTower ImageTower, ModelConfig Config, TagVocabulary Vocabulary, TagEmbeddingStore Embeddings, LogitScaleBias LogitScaleBias) Load(string directory, Device? device = null)
    {
        var config = JsonSerializer.Deserialize<ModelConfig>(File.ReadAllText(Path.Combine(directory, ConfigFileName)))
                     ?? throw new InvalidDataException($"'{directory}' does not contain a valid model config.");

        var imageTower = new ImageTower(config.EmbeddingDim, config.StemChannels, config.StageChannels, config.BlocksPerStage, device, config.FusedStages)
        {
            LocalizationFromFusedOnly = config.LocalizationFromFusedOnly
        };
        imageTower.LocalizationSkippedLaterals = config.LocalizationSkippedLaterals;
        imageTower.load(Path.Combine(directory, ImageTowerFileName));

        var vocabulary = TagVocabulary.Load(Path.Combine(directory, VocabularyFileName));
        var embeddings = TagEmbeddingStore.Load(Path.Combine(directory, EmbeddingsFileName));

        var logitScaleBias = new LogitScaleBias(device: device);
        var logitScaleBiasPath = Path.Combine(directory, LogitScaleBiasFileName);
        if (File.Exists(logitScaleBiasPath))
            logitScaleBias.load(logitScaleBiasPath);

        return (imageTower, config, vocabulary, embeddings, logitScaleBias);
    }
}
