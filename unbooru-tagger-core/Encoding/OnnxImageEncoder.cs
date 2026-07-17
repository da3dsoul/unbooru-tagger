using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace UnbooruTagger.Core.Encoding;

/// <summary>
/// Runs the image tower ONNX graph exported by the training CLI. The graph is expected
/// to expose two named outputs: the pooled global embedding and the pre-pool spatial
/// feature map (channels-first, i.e. [1, C, H, W]) — see CLAUDE.md's image-tower spec.
/// </summary>
public sealed class OnnxImageEncoder : IImageEncoder, IDisposable
{
    private readonly InferenceSession _session;
    private readonly int _inputSize;
    private readonly string _inputName;
    private readonly string _pooledOutputName;
    private readonly string _spatialOutputName;

    public int EmbeddingDim { get; }

    public OnnxImageEncoder(
        string modelPath,
        int embeddingDim,
        int inputSize = 224,
        string inputName = "pixel_values",
        string pooledOutputName = "pooled_embedding",
        string spatialOutputName = "spatial_features")
    {
        EmbeddingDim = embeddingDim;
        _inputSize = inputSize;
        _inputName = inputName;
        _pooledOutputName = pooledOutputName;
        _spatialOutputName = spatialOutputName;
        _session = new InferenceSession(modelPath);
    }

    public ImageEncoding Encode(string imagePath)
    {
        var flat = ImagePreprocessing.LoadAndNormalize(imagePath, _inputSize);
        var input = new DenseTensor<float>(flat, [1, 3, _inputSize, _inputSize]);

        using var results = _session.Run([NamedOnnxValue.CreateFromTensor(_inputName, input)]);

        var pooled = results.First(r => r.Name == _pooledOutputName).AsTensor<float>().ToArray();
        var spatial = results.First(r => r.Name == _spatialOutputName).AsTensor<float>();

        return new ImageEncoding(pooled, ToSpatialFeatureMap(spatial));
    }

    /// <summary>Converts the ONNX output's channels-first [1, C, H, W] layout to the channels-last layout <see cref="SpatialFeatureMap"/> indexes by.</summary>
    private static SpatialFeatureMap ToSpatialFeatureMap(Tensor<float> chw)
    {
        var channels = (int)chw.Dimensions[1];
        var height = (int)chw.Dimensions[2];
        var width = (int)chw.Dimensions[3];

        var hwc = new float[height * width * channels];
        for (var c = 0; c < channels; c++)
        for (var y = 0; y < height; y++)
        for (var x = 0; x < width; x++)
            hwc[(y * width + x) * channels + c] = chw[0, c, y, x];

        return new SpatialFeatureMap(hwc, height, width, channels);
    }

    public void Dispose() => _session.Dispose();
}
