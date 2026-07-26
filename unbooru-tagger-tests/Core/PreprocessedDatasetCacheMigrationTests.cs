using System.Text.Json;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Tests.Core;

public class PreprocessedDatasetCacheMigrationTests
{
    private const int LegacyFormatMagic = 0x4C425831; // "LBX1"

    /// <summary>Hand-writes a cache in the old fixed-stride, full-padded-canvas, float32 format — the layout PreprocessedDatasetCacheWriter no longer produces, but real pre-existing datasets are still in.</summary>
    private static void WriteLegacyCache(string directory, int inputSize, IReadOnlyList<EncodedImage> images, IReadOnlyList<IReadOnlyList<int>> tagRows)
    {
        Directory.CreateDirectory(directory);

        using (var stream = File.Create(Path.Combine(directory, "images.bin")))
        using (var writer = new BinaryWriter(stream))
        {
            writer.Write(LegacyFormatMagic);
            writer.Write(images.Count);
            writer.Write(inputSize);

            foreach (var image in images)
            {
                var padded = ImagePreprocessing.Reconstruct(image, inputSize);
                writer.Write(padded.Content.X);
                writer.Write(padded.Content.Y);
                writer.Write(padded.Content.Width);
                writer.Write(padded.Content.Height);
                foreach (var value in padded.Pixels)
                    writer.Write(value);
            }
        }

        using var labelWriter = new StreamWriter(Path.Combine(directory, "tag_rows.jsonl"));
        foreach (var rows in tagRows)
            labelWriter.WriteLine(JsonSerializer.Serialize(rows));
    }

    [Fact]
    public void ShrinkInPlace_ConvertsLegacyCacheToCurrentFormatLosslessly()
    {
        const int inputSize = 8;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            // One full-canvas image, one with real letterbox padding (content smaller than the canvas) — the padding must be dropped, not miscounted as content.
            var image0 = new EncodedImage(Enumerable.Range(0, 8 * 8 * 3).Select(i => (byte)(i % 256)).ToArray(), new LetterboxBox(0, 0, 8, 8));
            var image1 = new EncodedImage([10, 20, 30, 40, 50, 60], new LetterboxBox(X: 2, Y: 3, Width: 1, Height: 2));

            var tagRows = new List<IReadOnlyList<int>> { new[] { 0, 2 }, new[] { 1 } };
            WriteLegacyCache(directory, inputSize, [image0, image1], tagRows);

            var progressCalls = new List<(int Converted, int Total)>();
            PreprocessedDatasetCacheMigrator.ShrinkInPlace(directory, (converted, total) => progressCalls.Add((converted, total)));

            using var reader = new PreprocessedDatasetCacheReader(directory);
            Assert.Equal(2, reader.ImageCount);
            Assert.Equal(inputSize, reader.InputSize);
            Assert.Equal(new[] { 0, 2 }, reader.ImageTagRows[0]);
            Assert.Equal(new[] { 1 }, reader.ImageTagRows[1]);

            // Round-tripping through the legacy padded/normalized format and back must
            // reproduce exactly the same tensor a direct write of the new format would.
            Assert.Equal(ImagePreprocessing.Reconstruct(image0, inputSize).Pixels, reader.ReadImage(0).Pixels);
            Assert.Equal(ImagePreprocessing.Reconstruct(image1, inputSize).Pixels, reader.ReadImage(1).Pixels);
            Assert.Equal(image1.Content, reader.ReadImage(1).Content);

            Assert.Equal((2, 2), progressCalls[^1]);

            // No leftover backup or temp files once the swap is verified.
            Assert.False(File.Exists(Path.Combine(directory, "images.bin.lbx1.bak")));
            Assert.False(File.Exists(Path.Combine(directory, "tag_rows.jsonl.lbx1.bak")));
            Assert.False(Directory.Exists(Path.Combine(directory, ".shrink-tmp")));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void ShrinkInPlace_ResumesAfterCancellation()
    {
        const int inputSize = 4;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var images = Enumerable.Range(0, 5)
                .Select(i => new EncodedImage(Enumerable.Range(0, 4 * 4 * 3).Select(b => (byte)(b + i)).ToArray(), new LetterboxBox(0, 0, 4, 4)))
                .ToList();
            var tagRows = Enumerable.Range(0, 5).Select(i => (IReadOnlyList<int>)new List<int> { i }).ToList();
            WriteLegacyCache(directory, inputSize, images, tagRows);

            using (var cts = new CancellationTokenSource())
            {
                var converted = 0;
                Assert.Throws<OperationCanceledException>(() =>
                    PreprocessedDatasetCacheMigrator.ShrinkInPlace(directory, (done, _) =>
                    {
                        converted = done;
                        if (done == 2)
                            cts.Cancel();
                    }, cts.Token));
                Assert.Equal(2, converted);
            }

            // The original (legacy) cache must be untouched by the interrupted attempt.
            Assert.True(File.Exists(Path.Combine(directory, ".shrink-tmp", "images.bin")));

            PreprocessedDatasetCacheMigrator.ShrinkInPlace(directory);

            using var reader = new PreprocessedDatasetCacheReader(directory);
            Assert.Equal(5, reader.ImageCount);
            for (var i = 0; i < 5; i++)
                Assert.Equal(ImagePreprocessing.Reconstruct(images[i], inputSize).Pixels, reader.ReadImage(i).Pixels);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
