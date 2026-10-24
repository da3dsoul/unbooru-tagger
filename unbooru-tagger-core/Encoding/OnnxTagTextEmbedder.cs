using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace UnbooruTagger.Core.Encoding;

/// <summary>
/// Runs a small frozen text-embedding ONNX model (e.g. an exported sentence-embedding
/// model) to produce a tag's warm-start prior. Tokenization is supplied by the caller
/// rather than baked in here, since the right tokenizer (BPE/SentencePiece/etc.) is
/// whatever matches the specific frozen model chosen for this role.
/// </summary>
public sealed class OnnxTagTextEmbedder : ITagTextEmbedder, IDisposable
{
    private readonly InferenceSession _session;
    private readonly Func<string, long[]> _tokenize;
    private readonly string _inputIdsName;
    private readonly string _outputName;

    public int EmbeddingDim { get; }

    public OnnxTagTextEmbedder(
        string modelPath,
        int embeddingDim,
        Func<string, long[]> tokenize,
        string inputIdsName = "input_ids",
        string outputName = "embedding")
    {
        EmbeddingDim = embeddingDim;
        _tokenize = tokenize;
        _inputIdsName = inputIdsName;
        _outputName = outputName;
        _session = new InferenceSession(modelPath);
    }

    public float[] Embed(string tag)
    {
        var tokenIds = _tokenize(tag);
        var input = new DenseTensor<long>(tokenIds, [1, tokenIds.Length]);

        using var results = _session.Run([NamedOnnxValue.CreateFromTensor(_inputIdsName, input)]);
        var embedding = results.First(r => r.Name == _outputName).AsTensor<float>().ToArray();

        // The constructor takes EmbeddingDim as a plain parameter rather than reading it
        // off the graph (unlike OnnxImageEncoder's InputSize) since a text model's
        // output dim isn't reliably a static graph dimension the way an image model's
        // spatial input size is. Catching a mismatch here, right where it's produced,
        // gives a clear error instead of a cryptic shape-mismatch failure once this
        // vector is later used as a TagTower row.
        if (embedding.Length != EmbeddingDim)
            throw new InvalidDataException(
                $"'{_outputName}' produced a {embedding.Length}-dim embedding for tag '{tag}', but this embedder was constructed with EmbeddingDim={EmbeddingDim}.");

        return embedding;
    }

    public void Dispose() => _session.Dispose();
}
