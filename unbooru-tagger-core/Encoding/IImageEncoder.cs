namespace UnbooruTagger.Core.Encoding;

/// <summary>The image tower: produces a pooled embedding and a spatial feature map for one image.</summary>
public interface IImageEncoder
{
    int EmbeddingDim { get; }

    /// <summary>The square canvas size images are letterboxed into before encoding — needed to interpret <see cref="ImageEncoding.Content"/>.</summary>
    int InputSize { get; }

    ImageEncoding Encode(string imagePath);
}
