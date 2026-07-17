using SkiaSharp;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Data;

/// <summary>Loads and preprocesses images into a single NCHW float tensor batch, matching the normalization <c>OnnxImageEncoder</c> uses at inference time.</summary>
public static class ImageBatchLoader
{
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    public static Tensor Load(IReadOnlyList<string> imagePaths, int inputSize)
    {
        var channelSize = inputSize * inputSize;
        var imageSize = channelSize * 3;
        var flat = new float[imagePaths.Count * imageSize];

        for (var i = 0; i < imagePaths.Count; i++)
            FillOne(flat, i * imageSize, imagePaths[i], inputSize, channelSize);

        return tensor(flat, [imagePaths.Count, 3, inputSize, inputSize]);
    }

    private static void FillOne(float[] flat, int offset, string imagePath, int inputSize, int channelSize)
    {
        using var original = SKBitmap.Decode(imagePath)
            ?? throw new InvalidDataException($"Could not decode image at '{imagePath}'.");
        using var resized = original.Resize(new SKImageInfo(inputSize, inputSize), SKSamplingOptions.Default)
            ?? throw new InvalidOperationException($"Failed to resize image at '{imagePath}'.");

        for (var y = 0; y < inputSize; y++)
        for (var x = 0; x < inputSize; x++)
        {
            var pixel = resized.GetPixel(x, y);
            var pixelIndex = y * inputSize + x;
            flat[offset + (0 * channelSize) + pixelIndex] = (pixel.Red / 255f - Mean[0]) / Std[0];
            flat[offset + (1 * channelSize) + pixelIndex] = (pixel.Green / 255f - Mean[1]) / Std[1];
            flat[offset + (2 * channelSize) + pixelIndex] = (pixel.Blue / 255f - Mean[2]) / Std[2];
        }
    }
}
