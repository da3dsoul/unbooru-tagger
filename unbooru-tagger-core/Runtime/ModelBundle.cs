using UnbooruTagger.Core.Embedding;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Core.Runtime;

/// <summary>
/// The set of artifacts unbooru-tagger-training's `export-onnx` command writes and
/// unbooru-tagger-inference reads back, all under one directory: the image encoder
/// graph, the tag vocabulary, and the tag embedding table.
/// </summary>
public sealed class ModelBundle : IDisposable
{
    public const string ImageEncoderFileName = "image_encoder.onnx";
    public const string VocabularyFileName = "tag_vocabulary.json";
    public const string EmbeddingsFileName = "tag_embeddings.bin";

    public IImageEncoder ImageEncoder { get; }
    public TagVocabulary Vocabulary { get; }
    public TagEmbeddingStore Embeddings { get; }

    private readonly IDisposable? _ownedEncoder;

    private ModelBundle(IImageEncoder imageEncoder, TagVocabulary vocabulary, TagEmbeddingStore embeddings, IDisposable? ownedEncoder)
    {
        ImageEncoder = imageEncoder;
        Vocabulary = vocabulary;
        Embeddings = embeddings;
        _ownedEncoder = ownedEncoder;
    }

    public static ModelBundle Load(string directory, int inputSize = 224)
    {
        var vocabulary = TagVocabulary.Load(Path.Combine(directory, VocabularyFileName));
        var embeddings = TagEmbeddingStore.Load(Path.Combine(directory, EmbeddingsFileName));
        var encoder = new OnnxImageEncoder(Path.Combine(directory, ImageEncoderFileName), embeddings.EmbeddingDim, inputSize);
        return new ModelBundle(encoder, vocabulary, embeddings, encoder);
    }

    public void Dispose() => _ownedEncoder?.Dispose();
}
