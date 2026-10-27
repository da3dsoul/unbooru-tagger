using System.Threading.Channels;
using SkiaSharp;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Runtime;

namespace UnbooruTagger.Inference.Commands;

/// <summary>
/// Runs many images through the model in three overlapping stages, so the GPU (or the ONNX
/// Runtime CPU threads) never waits on image decoding and per-image post-processing never
/// waits on the model:
/// <list type="number">
/// <item><b>Load</b> (<c>workers</c> threads): decode, letterbox/normalize, and build the small
/// refinement guide <c>detect</c> needs. The full decoded bitmap is kept only when a caller
/// renders, so a huge image costs its decode once and then only its 512px input.</item>
/// <item><b>Encode</b> (one thread): <see cref="IImageEncoder.EncodeBatch"/> on up to
/// <c>batchSize</c> images at a time.</item>
/// <item><b>Process</b> (<c>workers</c> threads): the caller's per-image work (tag scoring,
/// heatmaps and boxes, rendering, output).</item>
/// </list>
/// The queues between stages are bounded, so memory stays flat however many images there are.
/// </summary>
public static class BatchPipeline
{
    public sealed record Loaded(string Path, PreprocessedImage Input, DetectCommandHandler.RefinementGuide? Guide, SKBitmap? Bitmap);

    public static readonly string[] SupportedExtensions = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"];

    public static List<string> FindImages(string directory, bool recursive) =>
        Directory.EnumerateFiles(directory, "*", recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly)
            .Where(path => SupportedExtensions.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase))
            .OrderBy(path => path, StringComparer.Ordinal)
            .ToList();

    /// <param name="withBoxes">Build each image's refinement guide and heatmap features (needed by <c>detect</c>).</param>
    /// <param name="keepBitmaps">Hand each image's decoded bitmap to <paramref name="process"/> (for rendering); it's disposed afterwards.</param>
    /// <param name="onFailure">An image that can't be decoded or processed; the run continues.</param>
    public static void Run(
        ModelBundle model, IReadOnlyList<string> paths, int batchSize, int workers, bool withBoxes, bool keepBitmaps,
        Action<Loaded, ImageEncoding> process, Action<string, Exception> onFailure)
    {
        var loaded = Channel.CreateBounded<Loaded>(new BoundedChannelOptions(batchSize * 2) { SingleReader = true });
        var encoded = Channel.CreateBounded<(Loaded, ImageEncoding)>(new BoundedChannelOptions(batchSize * 2) { SingleWriter = true });
        var inputSize = model.ImageEncoder.InputSize;

        var load = Task.Run(async () =>
        {
            try
            {
                await Parallel.ForEachAsync(paths, new ParallelOptions { MaxDegreeOfParallelism = workers }, async (path, ct) =>
                {
                    Loaded item;
                    try
                    {
                        var bitmap = SKBitmap.Decode(path) ?? throw new InvalidDataException($"Could not decode image at '{path}'.");
                        var input = ImagePreprocessing.Normalize(bitmap, inputSize);
                        var guide = withBoxes ? DetectCommandHandler.RefinementGuide.From(bitmap) : null;
                        if (!keepBitmaps)
                        {
                            bitmap.Dispose();
                            bitmap = null;
                        }
                        item = new Loaded(path, input, guide, bitmap);
                    }
                    catch (Exception e)
                    {
                        onFailure(path, e);
                        return;
                    }
                    await loaded.Writer.WriteAsync(item, ct);
                });
            }
            finally
            {
                loaded.Writer.Complete();
            }
        });

        var encode = Task.Run(async () =>
        {
            try
            {
                var batch = new List<Loaded>(batchSize);
                while (await loaded.Reader.WaitToReadAsync())
                {
                    // Fill the batch unless loading has finished; loading outpaces the model, so
                    // this rarely waits.
                    while (batch.Count < batchSize && (loaded.Reader.TryRead(out var next) || (await loaded.Reader.WaitToReadAsync() && loaded.Reader.TryRead(out next))))
                        batch.Add(next);
                    if (batch.Count == 0)
                        continue;
                    var encodings = model.ImageEncoder.EncodeBatch(batch.Select(b => b.Input).ToList(), withHeatmapFeatures: withBoxes);
                    for (var i = 0; i < batch.Count; i++)
                        await encoded.Writer.WriteAsync((batch[i], encodings[i]));
                    batch.Clear();
                }
            }
            finally
            {
                encoded.Writer.Complete();
            }
        });

        var post = Enumerable.Range(0, workers).Select(_ => Task.Run(async () =>
        {
            await foreach (var (item, encoding) in encoded.Reader.ReadAllAsync())
            {
                try
                {
                    process(item, encoding);
                }
                catch (Exception e)
                {
                    onFailure(item.Path, e);
                }
                finally
                {
                    item.Bitmap?.Dispose();
                }
            }
        })).ToArray();

        Task.WaitAll([load, encode, .. post]);
    }
}
