using UnbooruTagger.Core.Encoding;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Model;

/// <summary>
/// Builds a per-location validity mask for a batch's spatial feature map from each
/// image's <see cref="LetterboxBox"/>, so training can exclude the letterbox padding
/// bars (see <see cref="ImagePreprocessing"/>) from pooling and the localization loss
/// instead of letting a constant, content-free border silently drag on both. Wraps
/// <see cref="ImagePreprocessing.ComputeSpatialValidity"/> — the same rule
/// <see cref="UnbooruTagger.Core.Encoding.OnnxImageEncoder"/> uses to mask its own
/// pooling at inference, so training and inference agree on exactly what counts as padding.
/// </summary>
public static class SpatialMask
{
    /// <returns>A <c>[batch, 1, spatialHeight, spatialWidth]</c> tensor: 1 where that location's receptive field center falls inside the image's real content, 0 where it's letterbox padding.</returns>
    public static Tensor Build(IReadOnlyList<LetterboxBox> boxes, int inputSize, long spatialHeight, long spatialWidth, Device? device = null)
    {
        var height = (int)spatialHeight;
        var width = (int)spatialWidth;
        var mask = new float[boxes.Count * spatialHeight * spatialWidth];

        for (var b = 0; b < boxes.Count; b++)
        {
            var validity = ImagePreprocessing.ComputeSpatialValidity(boxes[b], inputSize, height, width);
            var imageOffset = b * spatialHeight * spatialWidth;

            for (var y = 0; y < height; y++)
            for (var x = 0; x < width; x++)
                if (validity[y, x])
                    mask[imageOffset + (y * spatialWidth) + x] = 1f;
        }

        var maskTensor = tensor(mask, [boxes.Count, 1, spatialHeight, spatialWidth]);
        return device is null ? maskTensor : maskTensor.to(device);
    }
}
