using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace UnbooruTagger.Core.Encoding;

/// <summary>Where <see cref="OnnxImageEncoder"/> runs the image tower.</summary>
public enum InferenceDevice
{
    Cpu,
    /// <summary>ONNX Runtime's CUDA provider on GPU 0; fails if it can't load.</summary>
    Cuda,
    /// <summary>CUDA when it loads, otherwise CPU.</summary>
    Auto,
}

/// <summary>
/// Runs the image tower ONNX graph exported by the training CLI. The graph is expected
/// to expose two named outputs: the pooled global embedding and the pre-pool spatial
/// feature map (channels-first, i.e. [batch, C, H, W]) — see CLAUDE.md's image-tower spec.
/// </summary>
public sealed class OnnxImageEncoder : IImageEncoder, IDisposable
{
    private readonly InferenceSession _session;
    private readonly int _inputSize;
    private readonly string _inputName;
    private readonly string _spatialOutputName;

    /// <summary>Optional output with the map heatmaps should use (see the training project's ImageTower.LocalizationFromFusedOnly).</summary>
    public const string LocalizationOutputName = "localization_features";

    public int EmbeddingDim { get; }
    public int InputSize => _inputSize;

    /// <summary>"CUDA" or "CPU": where the session actually ended up (<see cref="InferenceDevice.Auto"/> can fall back).</summary>
    public string Provider { get; }

    /// <summary>
    /// False for a bundle exported before the graph's batch dimension was symbolic: it only
    /// accepts one image per run, so <see cref="EncodeBatch"/> runs them one at a time.
    /// </summary>
    public bool SupportsBatches { get; }

    public OnnxImageEncoder(
        string modelPath,
        int embeddingDim,
        int inputSize = 224,
        string inputName = "pixel_values",
        string spatialOutputName = "spatial_features",
        InferenceDevice device = InferenceDevice.Cpu)
    {
        EmbeddingDim = embeddingDim;
        _inputName = inputName;
        _spatialOutputName = spatialOutputName;
        (_session, Provider) = CreateSession(modelPath, device);

        // The exported graph bakes in a fixed square resolution (see
        // ImageTowerOnnxExporter.Export), which is whatever --input-size the checkpoint
        // was trained/exported with — not necessarily the 224 default. Trust the graph's
        // own declared shape over the caller-supplied default so a model exported at a
        // different resolution just works instead of failing with a shape mismatch.
        var dimensions = _session.InputMetadata[_inputName].Dimensions;
        _inputSize = dimensions[2] > 0 ? dimensions[2] : inputSize;
        SupportsBatches = dimensions[0] <= 0;
    }

    private static (InferenceSession Session, string Provider) CreateSession(string modelPath, InferenceDevice device)
    {
        if (device != InferenceDevice.Cpu)
        {
            var options = new SessionOptions();
            try
            {
                options.AppendExecutionProvider_CUDA(0);
                return (new InferenceSession(modelPath, options), "CUDA");
            }
            catch (Exception e) when (device == InferenceDevice.Auto && e is OnnxRuntimeException or DllNotFoundException or EntryPointNotFoundException)
            {
                options.Dispose();
                Console.Error.WriteLine($"CUDA unavailable for ONNX Runtime ({e.Message.Split('\n')[0]}); running on CPU.");
            }
        }

        return (new InferenceSession(modelPath), "CPU");
    }

    public ImageEncoding Encode(string imagePath) =>
        EncodeBatch([ImagePreprocessing.LoadAndNormalize(imagePath, _inputSize)])[0];

    /// <summary>
    /// Encodes several preprocessed images in one run (one per run for a graph without a
    /// batch dimension). <paramref name="withHeatmapFeatures"/> false skips building each
    /// image's <see cref="ImageEncoding.SpatialFeatures"/>, for callers that only score whole-image tags.
    /// </summary>
    public IReadOnlyList<ImageEncoding> EncodeBatch(IReadOnlyList<PreprocessedImage> images, bool withHeatmapFeatures = true)
    {
        if (!SupportsBatches && images.Count > 1)
            return images.SelectMany(image => EncodeBatch([image], withHeatmapFeatures)).ToList();

        var imageSize = 3 * _inputSize * _inputSize;
        var pixels = new float[images.Count * imageSize];
        for (var i = 0; i < images.Count; i++)
            images[i].Pixels.CopyTo(pixels, i * imageSize);
        var input = new DenseTensor<float>(pixels, [images.Count, 3, _inputSize, _inputSize]);

        using var results = _session.Run([NamedOnnxValue.CreateFromTensor(_inputName, input)]);

        var spatial = results.First(r => r.Name == _spatialOutputName).AsTensor<float>();
        // A model trained with a separate localization map exports it as its own output;
        // heatmaps come from it, whole-image scores from the full map pooled below.
        var localization = results.FirstOrDefault(r => r.Name == LocalizationOutputName)?.AsTensor<float>();
        var heatmapSource = localization ?? spatial;

        var encodings = new ImageEncoding[images.Count];
        Parallel.For(0, images.Count, i =>
        {
            var pooled = MaskedPool(spatial, i, images[i].Content, _inputSize);
            var heatmapFeatures = withHeatmapFeatures ? ToSpatialFeatureMap(heatmapSource, i) : null;
            encodings[i] = new ImageEncoding(pooled, heatmapFeatures, images[i].Content);
        });
        return encodings;
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
    /// Reads image <paramref name="index"/>'s channels-first slice directly, in the same
    /// row-major location order as a channels-last pass, so the sums are identical.
    /// </summary>
    private static float[] MaskedPool(Tensor<float> chw, int index, LetterboxBox content, int canvasSize)
    {
        var (channels, height, width, slice) = Slice(chw, index);
        var validity = ImagePreprocessing.ComputeSpatialValidity(content, canvasSize, height, width);
        var valid = new List<int>(height * width);
        for (var y = 0; y < height; y++)
        for (var x = 0; x < width; x++)
            if (validity[y, x])
                valid.Add(y * width + x);

        // ComputeSpatialValidity always marks at least one location valid, so this never divides by zero.
        var sum = new float[channels];
        for (var c = 0; c < channels; c++)
        {
            var plane = slice.Span.Slice(c * height * width, height * width);
            var total = 0f;
            foreach (var location in valid)
                total += plane[location];
            sum[c] = total / valid.Count;
        }

        return sum;
    }

    /// <summary>Converts image <paramref name="index"/> of the ONNX output's channels-first [batch, C, H, W] layout to the channels-last layout <see cref="SpatialFeatureMap"/> indexes by.</summary>
    private static SpatialFeatureMap ToSpatialFeatureMap(Tensor<float> chw, int index)
    {
        var (channels, height, width, slice) = Slice(chw, index);
        var locations = height * width;
        var hwc = new float[locations * channels];
        for (var c = 0; c < channels; c++)
        {
            var plane = slice.Span.Slice(c * locations, locations);
            for (var p = 0; p < locations; p++)
                hwc[p * channels + c] = plane[p];
        }

        return new SpatialFeatureMap(hwc, height, width, channels);
    }

    private static (int Channels, int Height, int Width, ReadOnlyMemory<float> Slice) Slice(Tensor<float> chw, int index)
    {
        var channels = chw.Dimensions[1];
        var height = chw.Dimensions[2];
        var width = chw.Dimensions[3];
        var size = channels * height * width;
        var buffer = ((DenseTensor<float>)chw).Buffer;
        return (channels, height, width, buffer.Slice(index * size, size));
    }

    public void Dispose() => _session.Dispose();
}
