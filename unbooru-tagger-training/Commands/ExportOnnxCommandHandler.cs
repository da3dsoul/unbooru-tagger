using UnbooruTagger.Core.Runtime;
using UnbooruTagger.Core.Scoring;
using UnbooruTagger.Core.Vocabulary;
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
        var (imageTower, config, vocabulary, embeddings, logitScaleBias) = Checkpoint.Load(checkpointDir);

        Directory.CreateDirectory(modelDir);
        ImageTowerOnnxExporter.Export(imageTower, Path.Combine(modelDir, ModelBundle.ImageEncoderFileName), config.InputSize);
        vocabulary.Save(Path.Combine(modelDir, ModelBundle.VocabularyFileName));
        embeddings.Save(Path.Combine(modelDir, ModelBundle.EmbeddingsFileName));
        logitScaleBias.ToLogitScaling().Save(Path.Combine(modelDir, ModelBundle.LogitScalingFileName));

        // Optional companions written by `train`/`evaluate`: the exclusive-tag groups training
        // used, and per-tag thresholds calibrated on held-out data. Copied as-is when present.
        foreach (var optionalFile in new[] { ExclusiveTagGroups.FileName, TagThresholds.FileName, TagLocalizability.FileName })
        {
            var source = Path.Combine(checkpointDir, optionalFile);
            if (File.Exists(source))
                File.Copy(source, Path.Combine(modelDir, optionalFile), overwrite: true);
        }

        return 0;
    }
}
