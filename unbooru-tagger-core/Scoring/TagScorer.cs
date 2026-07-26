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
    /// contiguous above-threshold grid cells (4-connectivity) into boxes. <paramref name="heatmap"/>
    /// covers a <paramref name="canvasSize"/> x <paramref name="canvasSize"/> letterboxed
    /// canvas (padding bars included), so each box is first mapped from grid space to that
    /// canvas, then from <paramref name="content"/> (where the real image sits within it) to
    /// <paramref name="imageWidth"/> x <paramref name="imageHeight"/> pixels — the original
    /// image's own space. Approximate localization, not a trained detector — see CLAUDE.md.
    /// </summary>
    /// <param name="threshold">Absolute sigmoid floor — a tag whose whole heatmap sits below this never gets a box, regardless of <paramref name="relativePercentile"/>.</param>
    /// <param name="relativePercentile">
    /// Where to cut within THIS tag's own heatmap range, from 0 (its weakest cell) to 1 (only
    /// its single strongest cell). The effective cutoff is the higher of this and
    /// <paramref name="threshold"/>, so this can only tighten boxes, never loosen them — it
    /// keeps only the locations near a tag's own peak instead of everything that happens to
    /// clear the global absolute bar.
    /// </param>
    /// <param name="content">Where the original image's real content sits within the letterboxed canvas, in canvas pixel space (0..<paramref name="canvasSize"/>).</param>
    /// <param name="canvasSize">The side length of the square letterboxed canvas <paramref name="heatmap"/>'s grid covers.</param>
    public static List<BoundingBox> DetectBoxes(
        float[,] heatmap, float threshold, float relativePercentile,
        LetterboxBox content, int canvasSize, int imageWidth, int imageHeight)
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

        // Grid cell -> canvas pixel, then canvas pixel -> original image pixel: canvas
        // pixels inside `content` map linearly onto the original image; anything outside
        // it (the letterbox bars) clamps to the image's own edge instead of overshooting
        // past it or going negative.
        var gridToCanvasX = canvasSize / (float)width;
        var gridToCanvasY = canvasSize / (float)height;
        var canvasToImageX = imageWidth / (float)content.Width;
        var canvasToImageY = imageHeight / (float)content.Height;

        var boxes = new List<BoundingBox>();
        for (var y = 0; y < height; y++)
        for (var x = 0; x < width; x++)
        {
            if (visited[y, x] || heatmap[y, x] < effectiveThreshold)
                continue;

            boxes.Add(FloodFillComponent(
                heatmap, visited, x, y, effectiveThreshold,
                gridToCanvasX, gridToCanvasY, canvasToImageX, canvasToImageY, content, imageWidth, imageHeight));
        }

        return boxes;
    }

    /// <summary>BFS over 4-connected above-threshold grid cells starting at (startX, startY), returning their pixel-space bounding box.</summary>
    private static BoundingBox FloodFillComponent(
        float[,] heatmap, bool[,] visited, int startX, int startY, float threshold,
        float gridToCanvasX, float gridToCanvasY, float canvasToImageX, float canvasToImageY,
        LetterboxBox content, int imageWidth, int imageHeight)
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

        var canvasLeft = minX * gridToCanvasX;
        var canvasTop = minY * gridToCanvasY;
        var canvasRight = MathF.Ceiling((maxX + 1) * gridToCanvasX);
        var canvasBottom = MathF.Ceiling((maxY + 1) * gridToCanvasY);

        var pixelX = (int)Math.Clamp((canvasLeft - content.X) * canvasToImageX, 0, imageWidth);
        var pixelY = (int)Math.Clamp((canvasTop - content.Y) * canvasToImageY, 0, imageHeight);
        var pixelRight = (int)Math.Clamp(MathF.Ceiling((canvasRight - content.X) * canvasToImageX), 0, imageWidth);
        var pixelBottom = (int)Math.Clamp(MathF.Ceiling((canvasBottom - content.Y) * canvasToImageY), 0, imageHeight);

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
