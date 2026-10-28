using UnbooruTagger.Core.Scoring;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Held-out images' whole-image logits for every tag, for the evaluators. Either from pooled embeddings (one dot
/// product per pair, as every model before <see cref="Model.ClassificationPooling"/>), or a precomputed
/// [images × tags] matrix of raw scores when tags are pooled per tag and no single embedding gives them.
/// </summary>
public sealed class ImageTagLogits : IDisposable
{
    private readonly float[]? _pooled;
    private readonly float[]? _tagEmbeddings;
    private readonly int _dim;
    private readonly float[]? _raw;
    private Tensor? _deviceImages;
    private Tensor? _deviceTags;

    public LogitScaling Scaling { get; }
    public int ImageCount { get; }
    public int TagCount { get; }

    private ImageTagLogits(float[]? pooled, float[]? tagEmbeddings, int dim, float[]? raw, int imageCount, int tagCount, LogitScaling scaling)
    {
        (_pooled, _tagEmbeddings, _dim, _raw) = (pooled, tagEmbeddings, dim, raw);
        (ImageCount, TagCount, Scaling) = (imageCount, tagCount, scaling);
    }

    /// <param name="pooled">[images × dim] pooled embeddings.</param>
    /// <param name="tagEmbeddings">[tags × dim].</param>
    public static ImageTagLogits Linear(float[] pooled, float[] tagEmbeddings, int dim, LogitScaling scaling) =>
        new(pooled, tagEmbeddings, dim, null, pooled.Length / dim, tagEmbeddings.Length / dim, scaling);

    /// <param name="raw">[images × tags] raw scores, before <paramref name="scaling"/>.</param>
    public static ImageTagLogits Dense(float[] raw, int imageCount, LogitScaling scaling) =>
        new(null, null, 0, raw, imageCount, raw.Length / Math.Max(1, imageCount), scaling);

    public float Logit(int image, int tag)
    {
        if (_raw is not null)
            return Scaling.Apply(_raw[(long)image * TagCount + tag]);
        var a = new ReadOnlySpan<float>(_pooled, image * _dim, _dim);
        var b = new ReadOnlySpan<float>(_tagEmbeddings, tag * _dim, _dim);
        var sum = 0f;
        for (var i = 0; i < _dim; i++)
            sum += a[i] * b[i];
        return Scaling.Apply(sum);
    }

    public float Probability(int image, int tag) => 1f / (1f + MathF.Exp(-Logit(image, tag)));

    /// <summary>[images, tags.Length] scaled logits on <paramref name="device"/>.</summary>
    public Tensor Columns(int[] tags, Device device)
    {
        if (_raw is not null)
        {
            var values = new float[(long)ImageCount * tags.Length];
            Parallel.For(0, ImageCount, i =>
            {
                for (var j = 0; j < tags.Length; j++)
                    values[(long)i * tags.Length + j] = _raw[(long)i * TagCount + tags[j]];
            });
            return tensor(values, [ImageCount, tags.Length]).to(device) * Scaling.Scale + Scaling.Bias;
        }

        var rows = new float[tags.Length * _dim];
        for (var j = 0; j < tags.Length; j++)
            Array.Copy(_tagEmbeddings!, tags[j] * _dim, rows, j * _dim, _dim);
        return DeviceImages(device).matmul(tensor(rows, [tags.Length, _dim]).to(device).t()) * Scaling.Scale + Scaling.Bias;
    }

    /// <summary>[count, tags] scaled logits for images start..start+count on <paramref name="device"/>.</summary>
    public Tensor Rows(int start, int count, Device device)
    {
        if (_raw is not null)
            return tensor(_raw.AsSpan((int)((long)start * TagCount), count * TagCount).ToArray(), [count, TagCount]).to(device) * Scaling.Scale + Scaling.Bias;
        _deviceTags ??= tensor(_tagEmbeddings!, [TagCount, _dim]).to(device).DetachFromDisposeScope();
        var images = tensor(_pooled.AsSpan(start * _dim, count * _dim).ToArray(), [count, _dim]).to(device);
        return images.matmul(_deviceTags.t()) * Scaling.Scale + Scaling.Bias;
    }

    private Tensor DeviceImages(Device device) =>
        _deviceImages ??= tensor(_pooled!, [ImageCount, _dim]).to(device).DetachFromDisposeScope();

    public void Dispose()
    {
        _deviceImages?.Dispose();
        _deviceTags?.Dispose();
    }
}
