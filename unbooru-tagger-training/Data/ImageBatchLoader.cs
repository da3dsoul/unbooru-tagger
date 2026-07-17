using UnbooruTagger.Core.Encoding;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Data;

/// <summary>Loads a batch of images into a single NCHW float tensor via Core's shared <see cref="ImagePreprocessing"/>, so training sees exactly the normalization inference will use.</summary>
public static class ImageBatchLoader
{
    public static Tensor Load(IReadOnlyList<string> imagePaths, int inputSize)
    {
        var imageSize = 3 * inputSize * inputSize;
        var flat = new float[imagePaths.Count * imageSize];

        for (var i = 0; i < imagePaths.Count; i++)
            ImagePreprocessing.LoadAndNormalize(imagePaths[i], inputSize).CopyTo(flat, i * imageSize);

        return tensor(flat, [imagePaths.Count, 3, inputSize, inputSize]);
    }
}
