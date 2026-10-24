using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Training.Commands;

/// <summary>
/// Builds the optional <see cref="ITagTextEmbedder"/> for a new tag's warm-start prior
/// from the <c>--warm-start-model</c>/<c>--warm-start-vocab</c> CLI flags — the piece
/// CLAUDE.md's tag-tower section called out as existing in interface form
/// (<see cref="ITagTextEmbedder"/>/<see cref="OnnxTagTextEmbedder"/>) but with nothing
/// wired into it. Booru tags spell a multi-word concept with underscores
/// ("hatsune_miku_nurse_costume") where a text model expects spaces
/// ("hatsune miku nurse costume") — that normalization happens here, once, rather than
/// inside <see cref="WordPieceTokenizer"/>, which stays a generic text tokenizer with no
/// booru-specific knowledge.
/// </summary>
public static class TagWarmStartEmbedderFactory
{
    /// <returns><see langword="null"/> if neither flag was given (falls back to random-init, the pre-existing behavior). The caller owns disposing the result (concretely typed so a plain <c>using</c> works without an extra cast).</returns>
    public static OnnxTagTextEmbedder? Create(string? modelPath, string? vocabPath, int embeddingDim)
    {
        if (modelPath is null && vocabPath is null)
            return null;

        if (modelPath is null || vocabPath is null)
            throw new ArgumentException("--warm-start-model and --warm-start-vocab must be given together.");

        var tokenizer = WordPieceTokenizer.LoadVocabFile(vocabPath);
        return new OnnxTagTextEmbedder(modelPath, embeddingDim, tag => tokenizer.Tokenize(tag.Replace('_', ' ')));
    }
}
