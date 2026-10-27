using UnbooruTagger.Core.Embedding;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Scoring;
using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Core.Runtime;

/// <summary>
/// The set of artifacts unbooru-tagger-training's `export-onnx` command writes and
/// unbooru-tagger-inference reads back, all under one directory: the image encoder
/// graph, the tag vocabulary, the tag embedding table, and the learned logit
/// scale/bias — plus two optional files: <see cref="ExclusiveTagGroups.FileName"/>
/// (contradictory-tag suppression) and <see cref="TagThresholds.FileName"/> (per-tag
/// calibrated cutoffs from the training CLI's <c>evaluate</c> command). A bundle exported
/// before either existed loads with no exclusions and no per-tag thresholds.
/// </summary>
public sealed class ModelBundle : IDisposable
{
    public const string ImageEncoderFileName = "image_encoder.onnx";
    public const string VocabularyFileName = "tag_vocabulary.json";
    public const string EmbeddingsFileName = "tag_embeddings.bin";
    public const string LogitScalingFileName = "logit_scaling.json";

    public IImageEncoder ImageEncoder { get; }
    public TagVocabulary Vocabulary { get; }
    public TagEmbeddingStore Embeddings { get; }
    public LogitScaling LogitScaling { get; }
    public ExclusiveTagGroups ExclusiveGroups { get; }
    public TagThresholds? Thresholds { get; }
    /// <summary>Which tags get boxes in <c>detect</c>; the built-in defaults if the bundle has no <see cref="TagLocalizability.FileName"/>.</summary>
    public TagLocalizability Localizability { get; }

    private readonly IDisposable? _ownedEncoder;

    private ModelBundle(IImageEncoder imageEncoder, TagVocabulary vocabulary, TagEmbeddingStore embeddings, LogitScaling logitScaling, ExclusiveTagGroups exclusiveGroups, TagThresholds? thresholds, TagLocalizability localizability, IDisposable? ownedEncoder)
    {
        ExclusiveGroups = exclusiveGroups;
        Thresholds = thresholds;
        Localizability = localizability;
        ImageEncoder = imageEncoder;
        Vocabulary = vocabulary;
        Embeddings = embeddings;
        LogitScaling = logitScaling;
        _ownedEncoder = ownedEncoder;
    }

    public static ModelBundle Load(string directory, int inputSize = 224, InferenceDevice device = InferenceDevice.Cpu)
    {
        var vocabulary = TagVocabulary.Load(Path.Combine(directory, VocabularyFileName));
        var embeddings = TagEmbeddingStore.Load(Path.Combine(directory, EmbeddingsFileName));
        var logitScaling = LogitScaling.Load(Path.Combine(directory, LogitScalingFileName));
        var exclusiveGroups = ExclusiveTagGroups.Load(Path.Combine(directory, ExclusiveTagGroups.FileName), vocabulary, fallBackToDefaults: false);
        var thresholds = TagThresholds.Load(Path.Combine(directory, TagThresholds.FileName));
        var encoder = new OnnxImageEncoder(Path.Combine(directory, ImageEncoderFileName), embeddings.EmbeddingDim, inputSize, device: device);
        var localizability = TagLocalizability.Load(Path.Combine(directory, TagLocalizability.FileName));
        return new ModelBundle(encoder, vocabulary, embeddings, logitScaling, exclusiveGroups, thresholds, localizability, encoder);
    }

    /// <summary>
    /// The cutoff a tag must clear to be reported: an explicit <paramref name="overrideThreshold"/>
    /// (the CLI's <c>--threshold</c>) always wins; otherwise the tag's calibrated threshold if
    /// <see cref="Thresholds"/> has one; otherwise <paramref name="fallback"/>.
    /// </summary>
    public float ThresholdFor(int row, float? overrideThreshold, float fallback = 0.5f)
    {
        if (overrideThreshold is float explicitThreshold)
            return explicitThreshold;
        return Thresholds?.For(Vocabulary.GetByRowIndex(row).Tag, fallback) ?? fallback;
    }

    public void Dispose() => _ownedEncoder?.Dispose();
}
