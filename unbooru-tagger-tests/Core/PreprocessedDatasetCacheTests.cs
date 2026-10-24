using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Tests.Core;

public class PreprocessedDatasetCacheTests
{
    private static EncodedImage MakeImage(int width, int height, byte offset) =>
        new(
            Enumerable.Range(0, width * height * 3).Select(i => (byte)(i + offset)).ToArray(),
            new LetterboxBox(0, 0, width, height));

    [Fact]
    public void WriteAndRead_RoundTripsPixelsAndTagRows()
    {
        const int inputSize = 2;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var image0 = MakeImage(inputSize, inputSize, 0);
            var image1 = MakeImage(inputSize, inputSize, 100);

            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize))
            {
                writer.Append(image0, [0, 2]);
                writer.Append(image1, [1]);
            }

            using var reader = new PreprocessedDatasetCacheReader(directory);

            Assert.Equal(2, reader.ImageCount);
            Assert.Equal(inputSize, reader.InputSize);
            Assert.Equal(new[] { 0, 2 }, reader.ImageTagRows[0]);
            Assert.Equal(new[] { 1 }, reader.ImageTagRows[1]);
            Assert.Equal(ImagePreprocessing.Reconstruct(image0, inputSize).Pixels, reader.ReadImage(0).Pixels);
            Assert.Equal(image0.Content, reader.ReadImage(0).Content);
            Assert.Equal(ImagePreprocessing.Reconstruct(image1, inputSize).Pixels, reader.ReadImage(1).Pixels);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void ReadEncoded_MatchesReadImage_InRequestedOrder_WithDuplicatesAndTheLastRecord()
    {
        const int inputSize = 4;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            // Variable-size records (different content boxes), so offsets aren't a fixed stride.
            var images = new[]
            {
                MakeImage(4, 4, 0),
                new EncodedImage(Enumerable.Range(0, 4 * 2 * 3).Select(i => (byte)(i * 3)).ToArray(), new LetterboxBox(0, 1, 4, 2)),
                MakeImage(4, 4, 50),
                new EncodedImage(Enumerable.Range(0, 2 * 4 * 3).Select(i => (byte)(i * 7)).ToArray(), new LetterboxBox(1, 0, 2, 4)),
            };
            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize))
                foreach (var image in images)
                    writer.Append(image, [0]);

            using var reader = new PreprocessedDatasetCacheReader(directory);
            int[] requested = [3, 0, 3, 1, 2];
            var encoded = reader.ReadEncoded(requested, maxParallelism: 4);

            Assert.Equal(requested.Length, encoded.Length);
            for (var i = 0; i < requested.Length; i++)
            {
                var expected = reader.ReadImage(requested[i]);
                Assert.Equal(expected.Content, encoded[i].Content);
                Assert.Equal(expected.Pixels, ImagePreprocessing.Reconstruct(encoded[i], inputSize).Pixels);
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void WriteAndRead_PadsAroundContentSmallerThanCanvas()
    {
        // A 1x2 content region letterboxed into a 4x4 canvas: everything outside the
        // content box must reconstruct as exactly 0 (the neutral/pad value), and the
        // content pixels must land at exactly Content.X/Y, not at the origin.
        const int inputSize = 4;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var content = new LetterboxBox(X: 1, Y: 2, Width: 1, Height: 2);
            var image = new EncodedImage([10, 20, 30, 40, 50, 60], content);

            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize))
                writer.Append(image, [0]);

            using var reader = new PreprocessedDatasetCacheReader(directory);
            var expected = ImagePreprocessing.Reconstruct(image, inputSize).Pixels;
            var actual = reader.ReadImage(0).Pixels;

            Assert.Equal(expected, actual);

            var channelSize = inputSize * inputSize;
            // A canvas location outside the content box (e.g. the origin) must be exactly 0 padding.
            Assert.Equal(0f, actual[(0 * channelSize) + (0 * inputSize + 0)]);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Append_RejectsWrongPixelCount()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var writer = new PreprocessedDatasetCacheWriter(directory, inputSize: 4);
            Assert.Throws<ArgumentException>(() => writer.Append(new EncodedImage([1, 2], new LetterboxBox(0, 0, 4, 4)), []));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void MergeTagRows_OverwritesOnlyTheTargetedRows_AndSurvivesReopen()
    {
        const int inputSize = 2;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var image = MakeImage(inputSize, inputSize, 0);

            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize))
            {
                writer.Append(image, [0]);
                writer.Append(image, [1]);
                writer.Append(image, [2]);
                writer.Flush();

                Assert.Equal(
                    new[] { new[] { 0 }, new[] { 1 }, new[] { 2 } },
                    writer.ReadCommittedTagRows());

                // Row 1 turns out to also carry tag 7, discovered from a duplicate
                // crawled from the other site — row 0 and row 2 must be untouched.
                writer.MergeTagRows(new Dictionary<int, IReadOnlyList<int>> { [1] = [1, 7] });
            }

            using var reader = new PreprocessedDatasetCacheReader(directory);
            Assert.Equal(new[] { 0 }, reader.ImageTagRows[0]);
            Assert.Equal(new[] { 1, 7 }, reader.ImageTagRows[1]);
            Assert.Equal(new[] { 2 }, reader.ImageTagRows[2]);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Reader_WritesAnOffsetIndexSidecar_AndASecondOpenReadsBackTheSameImages()
    {
        // The regression this pins: PreprocessedImageIndex.LoadOrBuildOffsets must
        // produce byte-identical results whether it built the table itself (first open)
        // or loaded it from images.bin.offsets (every open after) — variable-width
        // records at three different sizes make sure a wrong cached offset would
        // actually corrupt a read instead of accidentally lining up.
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var image0 = MakeImage(2, 2, 0);
            var image1 = MakeImage(3, 1, 50);
            var image2 = MakeImage(1, 4, 100);

            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize: 4))
            {
                writer.Append(image0, [0]);
                writer.Append(image1, [1]);
                writer.Append(image2, [2]);
            }

            var offsetIndexPath = Path.Combine(directory, "images.bin.offsets");
            Assert.False(File.Exists(offsetIndexPath), "shouldn't exist before any reader has opened the cache");

            using (var firstReader = new PreprocessedDatasetCacheReader(directory))
            {
                Assert.Equal(image0.Content, firstReader.ReadImage(0).Content);
                Assert.Equal(image1.Content, firstReader.ReadImage(1).Content);
                Assert.Equal(image2.Content, firstReader.ReadImage(2).Content);
            }

            Assert.True(File.Exists(offsetIndexPath), "first open should have persisted the offset table it built");

            using var secondReader = new PreprocessedDatasetCacheReader(directory);
            Assert.Equal(ImagePreprocessing.Reconstruct(image0, 4).Pixels, secondReader.ReadImage(0).Pixels);
            Assert.Equal(ImagePreprocessing.Reconstruct(image1, 4).Pixels, secondReader.ReadImage(1).Pixels);
            Assert.Equal(ImagePreprocessing.Reconstruct(image2, 4).Pixels, secondReader.ReadImage(2).Pixels);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Reader_RebuildsTheOffsetIndex_WhenTheCacheGrewSinceItWasWritten()
    {
        // A stale index (wrong image count) must never be trusted -- silently reading
        // through it against a cache that's since grown would misread every offset
        // past whatever the stale table actually covered.
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize: 2))
                writer.Append(MakeImage(2, 2, 0), [0]);

            using (var firstReader = new PreprocessedDatasetCacheReader(directory))
                Assert.Equal(1, firstReader.ImageCount); // forces the (now stale) index to be written for count=1

            var image1 = MakeImage(2, 2, 100);
            using (var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(directory, inputSize: 2))
                writer.Append(image1, [1]);

            using var secondReader = new PreprocessedDatasetCacheReader(directory);
            Assert.Equal(2, secondReader.ImageCount);
            Assert.Equal(ImagePreprocessing.Reconstruct(image1, 2).Pixels, secondReader.ReadImage(1).Pixels);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Reader_FallsBackToRebuilding_WhenTheOffsetIndexIsCorrupt()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var image = MakeImage(2, 2, 0);
            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize: 2))
                writer.Append(image, [0]);

            using (var firstReader = new PreprocessedDatasetCacheReader(directory))
                _ = firstReader.ImageCount; // writes the sidecar

            File.WriteAllBytes(Path.Combine(directory, "images.bin.offsets"), [1, 2, 3]); // truncated garbage

            using var secondReader = new PreprocessedDatasetCacheReader(directory);
            Assert.Equal(ImagePreprocessing.Reconstruct(image, 2).Pixels, secondReader.ReadImage(0).Pixels);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void MergeTagRows_NoOp_WhenNothingToMerge()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var writer = new PreprocessedDatasetCacheWriter(directory, inputSize: 2);
            writer.Append(MakeImage(2, 2, 0), [0]);
            writer.Flush();

            // Must not touch (or close/reopen) the label file when there's nothing to merge.
            writer.MergeTagRows(new Dictionary<int, IReadOnlyList<int>>());
            writer.Append(MakeImage(2, 2, 0), [1]);

            Assert.Equal(2, writer.ImageCount);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
