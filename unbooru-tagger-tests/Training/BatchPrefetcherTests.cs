using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Training.Data;

namespace UnbooruTagger.Tests.Training;

public class BatchPrefetcherTests
{
    private const int InputSize = 2;

    // Image i is a full-canvas image whose every byte is i, so a decoded batch shows exactly which images it holds.
    private static EncodedImage[] ReadWindow(IReadOnlyList<int> ids) =>
        ids.Select(i => new EncodedImage(Enumerable.Repeat((byte)i, InputSize * InputSize * 3).ToArray(), new LetterboxBox(0, 0, InputSize, InputSize))).ToArray();

    [Fact]
    public void FiniteRequests_ComeBackInOrder_DecodedIntoTheirOwnSlots_ThenEnd()
    {
        var requests = Enumerable.Range(0, 7).Select(b => new BatchRequest([b * 2, b * 2 + 1])).ToList();
        using var prefetcher = new BatchPrefetcher(requests, ReadWindow, InputSize, windowBatches: 3, capacityBatches: 2);

        var floatsPerImage = 3 * InputSize * InputSize;
        for (var b = 0; b < requests.Count; b++)
        {
            Assert.True(prefetcher.TryTake(out var batch));
            Assert.Same(requests[b], batch.Request);
            for (var i = 0; i < 2; i++)
            {
                var expected = ImagePreprocessing.Reconstruct(ReadWindow([b * 2 + i])[0], InputSize).Pixels;
                Assert.Equal(expected, batch.Pixels.AsSpan(i * floatsPerImage, floatsPerImage).ToArray());
            }
        }

        Assert.False(prefetcher.TryTake(out _));
    }

    [Fact]
    public void InfiniteRequests_StopCleanlyOnDispose()
    {
        static IEnumerable<BatchRequest> Forever()
        {
            for (var i = 0; ; i++)
                yield return new BatchRequest([i % 200]);
        }

        var prefetcher = new BatchPrefetcher(Forever(), ReadWindow, InputSize, windowBatches: 4, capacityBatches: 4);
        for (var i = 0; i < 50; i++)
            Assert.True(prefetcher.TryTake(out _));
        prefetcher.Dispose();
    }

    [Fact]
    public void ABackgroundReadFailure_IsRethrownToTheConsumer()
    {
        var requests = new[] { new BatchRequest([0]), new BatchRequest([1]) };
        using var prefetcher = new BatchPrefetcher(requests, _ => throw new IOException("disk gone"), InputSize, windowBatches: 1);

        var error = Assert.Throws<InvalidOperationException>(() => prefetcher.TryTake(out _));
        Assert.IsType<IOException>(error.InnerException);
    }
}
