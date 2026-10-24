using System.Collections.Concurrent;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Training.Data;

/// <summary>A batch to load: dataset image indices, plus (for tag-centric training) why each image was sampled.</summary>
public sealed record BatchRequest(int[] Images, SampledImage[]? Sampled = null);

/// <summary>A loaded batch: NCHW pixels (normalized, padded) and each image's letterbox content box.</summary>
public sealed record LoadedBatch(BatchRequest Request, float[] Pixels, LetterboxBox[] Boxes);

/// <summary>
/// Loads batches ahead of the training loop on a background thread, so disk reads and
/// decoding overlap GPU compute instead of stalling it every step.
///
/// Requests are pulled <c>windowBatches</c> at a time; every image a window needs is read
/// in one <c>readWindow</c> call. With <see cref="Core.Dataset.PreprocessedDatasetCacheReader.ReadEncoded"/>,
/// that means one deduplicated, offset-sorted burst with many reads in flight — the access
/// pattern a striped spinning-disk array is fast at — instead of 20 one-at-a-time seeks
/// per step. Decoding into each batch's buffer then runs in parallel across images.
/// Up to <c>capacityBatches</c> finished batches wait in a bounded queue; the producer
/// blocks when it's full, which bounds memory (each 512px image is ~3MB as floats).
///
/// <c>requests</c> may be infinite (the training sampler) or finite (a validation pass);
/// <see cref="TryTake"/> returns false once a finite sequence is exhausted. A failure on
/// the background thread is rethrown to the consumer from <see cref="TryTake"/>.
/// </summary>
public sealed class BatchPrefetcher : IDisposable
{
    private readonly BlockingCollection<LoadedBatch> _ready;
    private readonly CancellationTokenSource _cancellation = new();
    private readonly Thread _producer;
    private Exception? _failure;

    public BatchPrefetcher(
        IEnumerable<BatchRequest> requests,
        Func<IReadOnlyList<int>, EncodedImage[]> readWindow,
        int inputSize,
        int windowBatches = 8,
        int capacityBatches = 16)
    {
        if (windowBatches < 1)
            throw new ArgumentException("windowBatches must be at least 1.", nameof(windowBatches));

        _ready = new BlockingCollection<LoadedBatch>(Math.Max(capacityBatches, windowBatches));
        _producer = new Thread(() => Produce(requests, readWindow, inputSize, windowBatches))
        {
            IsBackground = true,
            Name = nameof(BatchPrefetcher)
        };
        _producer.Start();
    }

    /// <summary>Total time the consumer spent blocked waiting for a batch — near zero when loading keeps up with training.</summary>
    public TimeSpan ConsumerWait { get; private set; }

    public bool TryTake(out LoadedBatch batch)
    {
        var started = System.Diagnostics.Stopwatch.GetTimestamp();
        try
        {
            foreach (var item in _ready.GetConsumingEnumerable(_cancellation.Token))
            {
                batch = item;
                return true;
            }
        }
        catch (OperationCanceledException)
        {
        }
        finally
        {
            ConsumerWait += System.Diagnostics.Stopwatch.GetElapsedTime(started);
        }

        if (_failure is not null)
            throw new InvalidOperationException("Background batch loading failed.", _failure);

        batch = null!;
        return false;
    }

    private void Produce(IEnumerable<BatchRequest> requests, Func<IReadOnlyList<int>, EncodedImage[]> readWindow, int inputSize, int windowBatches)
    {
        var imageFloats = 3 * inputSize * inputSize;
        try
        {
            using var enumerator = requests.GetEnumerator();
            var window = new List<BatchRequest>(windowBatches);
            while (!_cancellation.IsCancellationRequested)
            {
                window.Clear();
                while (window.Count < windowBatches && enumerator.MoveNext())
                    window.Add(enumerator.Current);
                if (window.Count == 0)
                    break;

                var allImages = window.SelectMany(r => r.Images).ToArray();
                var encoded = readWindow(allImages);

                var cursor = 0;
                foreach (var request in window)
                {
                    var count = request.Images.Length;
                    var pixels = new float[count * imageFloats];
                    var boxes = new LetterboxBox[count];
                    var first = cursor;
                    Parallel.For(0, count, i =>
                    {
                        var image = encoded[first + i];
                        boxes[i] = image.Content;
                        ImagePreprocessing.ReconstructInto(image, inputSize, pixels.AsSpan(i * imageFloats, imageFloats));
                    });
                    cursor += count;

                    _ready.Add(new LoadedBatch(request, pixels, boxes), _cancellation.Token);
                }
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception exception)
        {
            _failure = exception;
        }
        finally
        {
            _ready.CompleteAdding();
        }
    }

    public void Dispose()
    {
        _cancellation.Cancel();
        _producer.Join();
        _ready.Dispose();
        _cancellation.Dispose();
    }
}
