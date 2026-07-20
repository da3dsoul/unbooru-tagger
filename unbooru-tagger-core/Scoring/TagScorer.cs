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

    /// <summary>
    /// Rough bounding boxes for one tag: thresholds its <see cref="Heatmap"/> and groups
    /// contiguous above-threshold grid cells (4-connectivity) into boxes, each scaled from
    /// heatmap-grid space up to <paramref name="imageWidth"/> x <paramref name="imageHeight"/>
    /// pixels. Approximate localization, not a trained detector — see CLAUDE.md.
    /// </summary>
    /// <param name="threshold">Absolute sigmoid floor — a tag whose whole heatmap sits below this never gets a box, regardless of <paramref name="relativePercentile"/>.</param>
    /// <param name="relativePercentile">
    /// Where to cut within THIS tag's own heatmap range, from 0 (its weakest cell) to 1 (only
    /// its single strongest cell). The effective cutoff is the higher of this and
    /// <paramref name="threshold"/>, so this can only tighten boxes, never loosen them — it
    /// keeps only the locations near a tag's own peak instead of everything that happens to
    /// clear the global absolute bar.
    /// </param>
    public static List<BoundingBox> DetectBoxes(
        float[,] heatmap, float threshold, float relativePercentile, int imageWidth, int imageHeight)
    {
        var height = heatmap.GetLength(0);
        var width = heatmap.GetLength(1);

        var min = float.MaxValue;
        var max = float.MinValue;
        foreach (var value in heatmap)
        {
            min = Math.Min(min, value);
            max = Math.Max(max, value);
        }

        var effectiveThreshold = Math.Max(threshold, min + relativePercentile * (max - min));

        var visited = new bool[height, width];
        var scaleX = imageWidth / (float)width;
        var scaleY = imageHeight / (float)height;

        var boxes = new List<BoundingBox>();
        for (var y = 0; y < height; y++)
        for (var x = 0; x < width; x++)
        {
            if (visited[y, x] || heatmap[y, x] < effectiveThreshold)
                continue;

            boxes.Add(FloodFillComponent(heatmap, visited, x, y, effectiveThreshold, scaleX, scaleY));
        }

        return boxes;
    }

    /// <summary>BFS over 4-connected above-threshold grid cells starting at (startX, startY), returning their pixel-space bounding box.</summary>
    private static BoundingBox FloodFillComponent(
        float[,] heatmap, bool[,] visited, int startX, int startY, float threshold, float scaleX, float scaleY)
    {
        var height = heatmap.GetLength(0);
        var width = heatmap.GetLength(1);

        var queue = new Queue<(int X, int Y)>();
        queue.Enqueue((startX, startY));
        visited[startY, startX] = true;

        int minX = startX, maxX = startX, minY = startY, maxY = startY;
        var peakConfidence = heatmap[startY, startX];

        while (queue.TryDequeue(out var cell))
        {
            var (x, y) = cell;
            minX = Math.Min(minX, x);
            maxX = Math.Max(maxX, x);
            minY = Math.Min(minY, y);
            maxY = Math.Max(maxY, y);
            peakConfidence = Math.Max(peakConfidence, heatmap[y, x]);

            foreach (var (nx, ny) in new[] { (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1) })
            {
                if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                    continue;
                if (visited[ny, nx] || heatmap[ny, nx] < threshold)
                    continue;

                visited[ny, nx] = true;
                queue.Enqueue((nx, ny));
            }
        }

        var pixelX = (int)(minX * scaleX);
        var pixelY = (int)(minY * scaleY);
        var pixelRight = (int)MathF.Ceiling((maxX + 1) * scaleX);
        var pixelBottom = (int)MathF.Ceiling((maxY + 1) * scaleY);

        return new BoundingBox(pixelX, pixelY, pixelRight - pixelX, pixelBottom - pixelY, peakConfidence);
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
