using UnbooruTagger.Core.Encoding;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>A square crop window in localization-map cells: rows/columns [Y, Y + Size) x [X, X + Size).</summary>
public readonly record struct CellWindow(int Y, int X, int Size);

/// <summary>
/// Windows and crops for the crop-consistency loss
/// (<see cref="SigmoidContrastiveLoss.ComputeWindowConsistency"/>): the classifier, looking at
/// a crop alone, says whether a tag is inside that window, and the full image's localization
/// map pooled over the same window is trained to agree. Windows are square and aligned to
/// map cells, so the crop and the map region cover exactly the same pixels and resizing the
/// crop back to the input size doesn't distort it. They stay inside the image's letterboxed
/// content, so a crop never contains padding.
/// </summary>
public static class CropConsistency
{
    public static CellWindow[] SampleWindows(
        IReadOnlyList<LetterboxBox> boxes, int inputSize, int gridHeight, int gridWidth, Random random,
        float minFraction = 0.35f, float maxFraction = 0.7f)
    {
        var windows = new CellWindow[boxes.Count];
        for (var b = 0; b < boxes.Count; b++)
        {
            var validity = ImagePreprocessing.ComputeSpatialValidity(boxes[b], inputSize, gridHeight, gridWidth);
            int minY = gridHeight, maxY = -1, minX = gridWidth, maxX = -1;
            for (var y = 0; y < gridHeight; y++)
            for (var x = 0; x < gridWidth; x++)
            {
                if (!validity[y, x])
                    continue;
                minY = Math.Min(minY, y);
                maxY = Math.Max(maxY, y);
                minX = Math.Min(minX, x);
                maxX = Math.Max(maxX, x);
            }

            var height = maxY - minY + 1;
            var width = maxX - minX + 1;
            var shortSide = Math.Min(height, width);
            var fraction = minFraction + (float)random.NextDouble() * (maxFraction - minFraction);
            var size = Math.Clamp((int)MathF.Round(fraction * shortSide), 1, shortSide);
            windows[b] = new CellWindow(
                minY + random.Next(height - size + 1),
                minX + random.Next(width - size + 1),
                size);
        }

        return windows;
    }

    /// <summary>Each image's window, cut from <paramref name="pixels"/> ([B, 3, S, S]) and resized back to S x S.</summary>
    public static Tensor CropPixels(Tensor pixels, IReadOnlyList<CellWindow> windows, int gridSize)
    {
        var inputSize = (int)pixels.shape[2];
        var stride = inputSize / gridSize;
        var crops = new Tensor[windows.Count];
        for (var b = 0; b < windows.Count; b++)
        {
            var w = windows[b];
            var crop = pixels[b..(b + 1), .., (w.Y * stride)..((w.Y + w.Size) * stride), (w.X * stride)..((w.X + w.Size) * stride)];
            crops[b] = nn.functional.interpolate(crop, size: [inputSize, inputSize], mode: InterpolationMode.Bilinear, align_corners: false);
        }

        return cat(crops, 0);
    }

    /// <summary>[B, 1, H, W] mask: 1 inside each image's window.</summary>
    public static Tensor WindowMask(IReadOnlyList<CellWindow> windows, int gridHeight, int gridWidth, Device? device = null)
    {
        var mask = new float[windows.Count * gridHeight * gridWidth];
        for (var b = 0; b < windows.Count; b++)
        {
            var w = windows[b];
            for (var y = w.Y; y < w.Y + w.Size; y++)
            for (var x = w.X; x < w.X + w.Size; x++)
                mask[(b * gridHeight + y) * gridWidth + x] = 1f;
        }

        var result = tensor(mask, [windows.Count, 1, gridHeight, gridWidth]);
        return device is null ? result : result.to(device);
    }
}
