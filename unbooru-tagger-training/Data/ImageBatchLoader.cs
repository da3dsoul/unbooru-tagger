using UnbooruTagger.Core.Encoding;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Data;

/// <summary>Loads a batch of images into a single NCHW float tensor via Core's shared <see cref="ImagePreprocessing"/>, so training sees exactly the normalization inference will use.</summary>
public static class ImageBatchLoader
{
    /// <returns>The batch's pixel tensor, plus each image's letterbox content box (in <paramref name="inputSize"/> canvas space) — needed to mask out padding when pooling/scoring against the spatial feature map.</returns>
    public static (Tensor Pixels, IReadOnlyList<LetterboxBox> Boxes) Load(IReadOnlyList<string> imagePaths, int inputSize)
    {
        var imageSize = 3 * inputSize * inputSize;
        var flat = new float[imagePaths.Count * imageSize];
        var boxes = new LetterboxBox[imagePaths.Count];

        for (var i = 0; i < imagePaths.Count; i++)
        {
            var preprocessed = ImagePreprocessing.LoadAndNormalize(imagePaths[i], inputSize);
            preprocessed.Pixels.CopyTo(flat, i * imageSize);
            boxes[i] = preprocessed.Content;
        }

        return (tensor(flat, [imagePaths.Count, 3, inputSize, inputSize]), boxes);
    }
}
