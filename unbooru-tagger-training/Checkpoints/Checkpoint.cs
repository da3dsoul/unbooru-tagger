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

    /// <summary>Whether <paramref name="directory"/> holds a complete checkpoint <see cref="Load"/> can read back.</summary>
    public static bool Exists(string directory) =>
        File.Exists(Path.Combine(directory, ConfigFileName)) &&
        File.Exists(Path.Combine(directory, ImageTowerFileName)) &&
        File.Exists(Path.Combine(directory, VocabularyFileName)) &&
        File.Exists(Path.Combine(directory, EmbeddingsFileName));

    public static void Save(string directory, ImageTower imageTower, ModelConfig config, TagVocabulary vocabulary, TagEmbeddingStore embeddings)
    {
        Directory.CreateDirectory(directory);
        imageTower.save(Path.Combine(directory, ImageTowerFileName));
        File.WriteAllText(Path.Combine(directory, ConfigFileName), JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true }));
        vocabulary.Save(Path.Combine(directory, VocabularyFileName));
        embeddings.Save(Path.Combine(directory, EmbeddingsFileName));
    }

    public static (ImageTower ImageTower, ModelConfig Config, TagVocabulary Vocabulary, TagEmbeddingStore Embeddings) Load(string directory, Device? device = null)
    {
        var config = JsonSerializer.Deserialize<ModelConfig>(File.ReadAllText(Path.Combine(directory, ConfigFileName)))
                     ?? throw new InvalidDataException($"'{directory}' does not contain a valid model config.");

        var imageTower = new ImageTower(config.EmbeddingDim, config.StemChannels, config.StageChannels, config.BlocksPerStage, device);
        imageTower.load(Path.Combine(directory, ImageTowerFileName));

        var vocabulary = TagVocabulary.Load(Path.Combine(directory, VocabularyFileName));
        var embeddings = TagEmbeddingStore.Load(Path.Combine(directory, EmbeddingsFileName));

        return (imageTower, config, vocabulary, embeddings);
    }
}
