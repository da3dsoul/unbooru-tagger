namespace UnbooruTagger.Core.Encoding;

/// <summary>The image tower: produces a pooled embedding and a spatial feature map for one image.</summary>
public interface IImageEncoder
{
    int EmbeddingDim { get; }

    ImageEncoding Encode(string imagePath);
}
