using SkiaSharp;

namespace UnbooruTagger.Core.Scoring;

/// <summary>
/// Sharpens a coarse localization <see cref="TagScorer.Heatmap"/> by snapping it to the
/// original image's edges — a joint/cross bilateral filter using the image itself as
/// guidance. This is the practical, label-free version of the CAM+DenseCRF refinement
/// trick from weakly-supervised segmentation: it pulls a blobby heatmap boundary in to
/// match the nearest real color edge, without any bounding-box training data.
/// </summary>
public static class HeatmapRefiner
{
    /// <summary>
    /// Resizes <paramref name="original"/> down to <paramref name="size"/> x <paramref name="size"/>
    /// (matching how <see cref="Encoding.ImagePreprocessing"/> squares off the image for the
    /// model) and extracts it as an RGB guide for <see cref="Refine"/>. Build this once per
    /// image and reuse it across every tag's heatmap — it doesn't depend on which tag is
    /// being refined.
    /// </summary>
    public static float[,,] BuildGuide(SKBitmap original, int size)
    {
        var resizeInfo = new SKImageInfo(size, size, SKColorType.Rgba8888, SKAlphaType.Unpremul);
        using var resized = original.Resize(resizeInfo, SKSamplingOptions.Default)
            ?? throw new InvalidOperationException("Failed to resize image for heatmap refinement.");

        var guide = new float[size, size, 3];
        var pixels = resized.GetPixelSpan();
        var rowBytes = resized.RowBytes;
        const int bytesPerPixel = 4;

        for (var y = 0; y < size; y++)
        {
            var rowStart = y * rowBytes;
            for (var x = 0; x < size; x++)
            {
                var offset = rowStart + x * bytesPerPixel;
                guide[y, x, 0] = pixels[offset] / 255f;
                guide[y, x, 1] = pixels[offset + 1] / 255f;
                guide[y, x, 2] = pixels[offset + 2] / 255f;
            }
        }

        return guide;
    }

    /// <summary>
    /// Upsamples <paramref name="heatmap"/> to the guide's resolution (bilinear) and refines it
    /// with a joint bilateral filter: each output cell is a weighted average of nearby heatmap
    /// values, weighted by both spatial distance and color similarity to the guide image at
    /// that location. Cells across a sharp color edge barely influence each other, so the
    /// heatmap's soft blob edge collapses onto the nearest real object boundary.
    /// </summary>
    public static float[,] Refine(
        float[,] heatmap, float[,,] guide, int radius = 4, float spatialSigma = 3f, float colorSigma = 0.15f)
    {
        var size = guide.GetLength(0);
        var upsampled = UpsampleBilinear(heatmap, size);
        var refined = new float[size, size];

        // Depends only on (dx, dy), so compute it once instead of per output pixel.
        var spatialWeights = new float[2 * radius + 1, 2 * radius + 1];
        for (var dy = -radius; dy <= radius; dy++)
        for (var dx = -radius; dx <= radius; dx++)
            spatialWeights[dy + radius, dx + radius] =
                MathF.Exp(-(dx * dx + dy * dy) / (2f * spatialSigma * spatialSigma));

        for (var y = 0; y < size; y++)
        for (var x = 0; x < size; x++)
        {
            var r0 = guide[y, x, 0];
            var g0 = guide[y, x, 1];
            var b0 = guide[y, x, 2];

            var weightedSum = 0f;
            var weightTotal = 0f;

            for (var dy = -radius; dy <= radius; dy++)
            {
                var ny = y + dy;
                if (ny < 0 || ny >= size)
                    continue;

                for (var dx = -radius; dx <= radius; dx++)
                {
                    var nx = x + dx;
                    if (nx < 0 || nx >= size)
                        continue;

                    var dr = guide[ny, nx, 0] - r0;
                    var dg = guide[ny, nx, 1] - g0;
                    var db = guide[ny, nx, 2] - b0;
                    var colorDistanceSq = dr * dr + dg * dg + db * db;
                    var colorWeight = MathF.Exp(-colorDistanceSq / (2f * colorSigma * colorSigma));

                    var weight = spatialWeights[dy + radius, dx + radius] * colorWeight;
                    weightedSum += weight * upsampled[ny, nx];
                    weightTotal += weight;
                }
            }

            refined[y, x] = weightedSum / weightTotal;
        }

        return refined;
    }

    private static float[,] UpsampleBilinear(float[,] source, int targetSize)
    {
        var sourceHeight = source.GetLength(0);
        var sourceWidth = source.GetLength(1);
        var result = new float[targetSize, targetSize];

        for (var y = 0; y < targetSize; y++)
        {
            var sy = (y + 0.5f) * sourceHeight / targetSize - 0.5f;
            var y0 = (int)MathF.Floor(sy);
            var yFrac = sy - y0;

            for (var x = 0; x < targetSize; x++)
            {
                var sx = (x + 0.5f) * sourceWidth / targetSize - 0.5f;
                var x0 = (int)MathF.Floor(sx);
                var xFrac = sx - x0;

                var v00 = SampleClamped(source, x0, y0, sourceWidth, sourceHeight);
                var v10 = SampleClamped(source, x0 + 1, y0, sourceWidth, sourceHeight);
                var v01 = SampleClamped(source, x0, y0 + 1, sourceWidth, sourceHeight);
                var v11 = SampleClamped(source, x0 + 1, y0 + 1, sourceWidth, sourceHeight);

                var top = v00 + (v10 - v00) * xFrac;
                var bottom = v01 + (v11 - v01) * xFrac;
                result[y, x] = top + (bottom - top) * yFrac;
            }
        }

        return result;
    }

    private static float SampleClamped(float[,] source, int x, int y, int width, int height)
    {
        x = Math.Clamp(x, 0, width - 1);
        y = Math.Clamp(y, 0, height - 1);
        return source[y, x];
    }
}
