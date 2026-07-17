using UnbooruTagger.Core.Runtime;
using UnbooruTagger.Training.Checkpoints;
using UnbooruTagger.Training.Export;

namespace UnbooruTagger.Training.Commands;

/// <summary>
/// Exports a training checkpoint into the ONNX + vocabulary + embeddings bundle
/// unbooru-tagger-inference reads (see <see cref="ModelBundle"/>).
/// </summary>
public static class ExportOnnxCommandHandler
{
    public static int Run(string checkpointDir, string modelDir)
    {
        var (imageTower, config, vocabulary, embeddings) = Checkpoint.Load(checkpointDir);

        Directory.CreateDirectory(modelDir);
        ImageTowerOnnxExporter.Export(imageTower, Path.Combine(modelDir, ModelBundle.ImageEncoderFileName), config.InputSize);
        vocabulary.Save(Path.Combine(modelDir, ModelBundle.VocabularyFileName));
        embeddings.Save(Path.Combine(modelDir, ModelBundle.EmbeddingsFileName));

        return 0;
    }
}
