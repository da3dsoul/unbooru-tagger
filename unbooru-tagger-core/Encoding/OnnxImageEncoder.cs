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
    private readonly string _spatialOutputName;

    public int EmbeddingDim { get; }
    public int InputSize => _inputSize;

    public OnnxImageEncoder(
        string modelPath,
        int embeddingDim,
        int inputSize = 224,
        string inputName = "pixel_values",
        string spatialOutputName = "spatial_features")
    {
        EmbeddingDim = embeddingDim;
        _inputSize = inputSize;
        _inputName = inputName;
        _spatialOutputName = spatialOutputName;
        _session = new InferenceSession(modelPath);
    }

    public ImageEncoding Encode(string imagePath)
    {
        var preprocessed = ImagePreprocessing.LoadAndNormalize(imagePath, _inputSize);
        var input = new DenseTensor<float>(preprocessed.Pixels, [1, 3, _inputSize, _inputSize]);

        using var results = _session.Run([NamedOnnxValue.CreateFromTensor(_inputName, input)]);

        var spatial = results.First(r => r.Name == _spatialOutputName).AsTensor<float>();
        var spatialFeatures = ToSpatialFeatureMap(spatial);
        var pooled = MaskedPool(spatialFeatures, preprocessed.Content, _inputSize);

        return new ImageEncoding(pooled, spatialFeatures, preprocessed.Content);
    }

    /// <summary>
    /// Global average pool restricted to locations inside <paramref name="content"/>,
    /// instead of the ONNX graph's own <c>pooled_embedding</c> output, which averages
    /// every location uniformly including the letterbox padding bars. Training's masked
    /// pooling (see <c>UnbooruTagger.Training.Model.ImageTower.MaskedPool</c> and
    /// <c>SpatialMask</c>) never gave the image tower gradient signal for what those
    /// padding locations should contribute once pooled, so leaving them in here would
    /// silently dilute the embedding relative to what the model was actually trained to
    /// produce. Recomputed here from the spatial output instead of changing what the
    /// exported graph itself exposes, so the .onnx file's own documented two-output
    /// contract (see this class's summary) is unaffected for any other consumer of it.
    /// </summary>
    private static float[] MaskedPool(SpatialFeatureMap spatial, LetterboxBox content, int canvasSize)
    {
        var validity = ImagePreprocessing.ComputeSpatialValidity(content, canvasSize, spatial.Height, spatial.Width);

        var sum = new float[spatial.Channels];
        var validCount = 0;
        for (var y = 0; y < spatial.Height; y++)
        for (var x = 0; x < spatial.Width; x++)
        {
            if (!validity[y, x])
                continue;

            var vector = spatial[y, x];
            for (var c = 0; c < vector.Length; c++)
                sum[c] += vector[c];
            validCount++;
        }

        // ComputeSpatialValidity always marks at least one location valid, so this never divides by zero.
        for (var c = 0; c < sum.Length; c++)
            sum[c] /= validCount;

        return sum;
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
