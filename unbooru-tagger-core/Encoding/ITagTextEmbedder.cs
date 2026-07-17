namespace UnbooruTagger.Core.Encoding;

/// <summary>
/// Computes the warm-start prior vector for a tag string using a small frozen
/// language model, so a brand-new tag's row starts out semantically close to
/// related tags instead of at random (see CLAUDE.md's tag-tower spec).
/// </summary>
public interface ITagTextEmbedder
{
    int EmbeddingDim { get; }

    float[] Embed(string tag);
}
