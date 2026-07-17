using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Core.Scoring;

/// <summary>
/// SigLIP-style sigmoid scoring between image and tag embeddings: every image-tag
/// pair is an independent match/no-match decision (see CLAUDE.md's training objective).
/// </summary>
public static class TagScorer
{
    /// <summary>Confidence that <paramref name="tagEmbedding"/> applies to the whole image.</summary>
    public static float Score(ReadOnlySpan<float> imageEmbedding, ReadOnlySpan<float> tagEmbedding) =>
        Sigmoid(Dot(imageEmbedding, tagEmbedding));

    /// <summary>
    /// A rough localization heatmap for one tag: sigmoid(dot(tag, spatial location)) at
    /// every position in the pre-pool feature map. No CAM/Grad-CAM step needed — this
    /// falls directly out of the dual-encoder design (MaskCLIP-style).
    /// </summary>
    public static float[,] Heatmap(ReadOnlySpan<float> tagEmbedding, SpatialFeatureMap spatialFeatures)
    {
        var heatmap = new float[spatialFeatures.Height, spatialFeatures.Width];
        for (var y = 0; y < spatialFeatures.Height; y++)
        for (var x = 0; x < spatialFeatures.Width; x++)
            heatmap[y, x] = Sigmoid(Dot(tagEmbedding, spatialFeatures[y, x]));

        return heatmap;
    }

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must be the same length to dot-product.");

        var sum = 0f;
        for (var i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
