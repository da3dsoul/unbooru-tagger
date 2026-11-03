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
    /// <param name="ignorePooling">
    /// Export a checkpoint that scores localizable tags per tag (classification pooling or glimpses) anyway. Its heatmaps
    /// and boxes are right, since they come from the localization map, but the bundle scores those tags from the averaged
    /// embedding, which the model wasn't trained to score them with: pair it with <c>render-gold --scores</c>.
    /// </param>
    public static int Run(string checkpointDir, string modelDir, bool ignorePooling = false)
    {
        var (imageTower, config, vocabulary, embeddings, logitScaleBias) = Checkpoint.Load(checkpointDir);
        // Inference scores from the averaged embedding only; a per-tag pooled classifier would be scored wrong.
        if (config.ClassificationPoolingTemperature > 0 && !ignorePooling)
            throw new NotSupportedException(
                $"This checkpoint scores localizable tags per tag (classification pooling temperature {config.ClassificationPoolingTemperature}), which the ONNX bundle and inference don't support yet (--ignore-pooling exports it for its boxes).");

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
