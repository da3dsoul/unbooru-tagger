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
    // ImageNet normalization stats — a reasonable default for ViT/ConvNeXt backbones;
    // override if the trained encoder used different preprocessing.
    private static readonly float[] DefaultMean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] DefaultStd = [0.229f, 0.224f, 0.225f];

    private readonly InferenceSession _session;
    private readonly int _inputSize;
    private readonly string _inputName;
    private readonly string _pooledOutputName;
    private readonly string _spatialOutputName;
    private readonly float[] _mean;
    private readonly float[] _std;

    public int EmbeddingDim { get; }

    public OnnxImageEncoder(
        string modelPath,
        int embeddingDim,
        int inputSize = 224,
        string inputName = "pixel_values",
        string pooledOutputName = "pooled_embedding",
        string spatialOutputName = "spatial_features",
        float[]? mean = null,
        float[]? std = null)
    {
        EmbeddingDim = embeddingDim;
        _inputSize = inputSize;
        _inputName = inputName;
        _pooledOutputName = pooledOutputName;
        _spatialOutputName = spatialOutputName;
        _mean = mean ?? DefaultMean;
        _std = std ?? DefaultStd;
        _session = new InferenceSession(modelPath);
    }

    public ImageEncoding Encode(string imagePath)
    {
        using var original = SKBitmap.Decode(imagePath)
            ?? throw new InvalidDataException($"Could not decode image at '{imagePath}'.");
        using var resized = original.Resize(new SKImageInfo(_inputSize, _inputSize), SKSamplingOptions.Default)
            ?? throw new InvalidOperationException($"Failed to resize image at '{imagePath}'.");

        var input = new DenseTensor<float>([1, 3, _inputSize, _inputSize]);
        for (var y = 0; y < _inputSize; y++)
        for (var x = 0; x < _inputSize; x++)
        {
            var pixel = resized.GetPixel(x, y);
            input[0, 0, y, x] = (pixel.Red / 255f - _mean[0]) / _std[0];
            input[0, 1, y, x] = (pixel.Green / 255f - _mean[1]) / _std[1];
            input[0, 2, y, x] = (pixel.Blue / 255f - _mean[2]) / _std[2];
        }

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
