using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Tests.Core;

public class PreprocessedDatasetCacheCompactorTests
{
    private static EncodedImage MakeImage(int seed, int width = 3, int height = 2) =>
        new(Enumerable.Range(0, width * height * 3).Select(b => (byte)(b + seed)).ToArray(), new LetterboxBox(0, 0, width, height));

    [Fact]
    public void DropRows_RemovesRowsAndShiftsSurvivorsDownWithACorrectIndexMap()
    {
        const int inputSize = 8;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var images = Enumerable.Range(0, 5).Select(i => MakeImage(i)).ToList();
            var tagRows = Enumerable.Range(0, 5).Select(i => (IReadOnlyList<int>)new List<int> { i, i + 100 }).ToList();

            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize))
            {
                for (var i = 0; i < images.Count; i++)
                    writer.Append(images[i], tagRows[i]);
            }

            // Drop rows 1 and 3 -- survivors are 0, 2, 4, which must land at new indices 0, 1, 2 respectively.
            var result = PreprocessedDatasetCacheCompactor.DropRows(directory, new HashSet<int> { 1, 3 });

            Assert.Equal(5, result.OriginalCount);
            Assert.Equal(3, result.KeptCount);
            Assert.Equal(new Dictionary<int, int> { [0] = 0, [2] = 1, [4] = 2 }, result.OldToNewIndex);

            using var reader = new PreprocessedDatasetCacheReader(directory);
            Assert.Equal(3, reader.ImageCount);

            Assert.Equal(ImagePreprocessing.Reconstruct(images[0], inputSize).Pixels, reader.ReadImage(0).Pixels);
            Assert.Equal(ImagePreprocessing.Reconstruct(images[2], inputSize).Pixels, reader.ReadImage(1).Pixels);
            Assert.Equal(ImagePreprocessing.Reconstruct(images[4], inputSize).Pixels, reader.ReadImage(2).Pixels);

            Assert.Equal(tagRows[0], reader.ImageTagRows[0]);
            Assert.Equal(tagRows[2], reader.ImageTagRows[1]);
            Assert.Equal(tagRows[4], reader.ImageTagRows[2]);

            // No leftover backup or temp files once the swap is verified.
            Assert.False(File.Exists(Path.Combine(directory, "images.bin.pre-compact.bak")));
            Assert.False(File.Exists(Path.Combine(directory, "tag_rows.sqlite.pre-compact.bak")));
            Assert.False(Directory.Exists(Path.Combine(directory, ".compact-tmp")));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void DropRows_DroppingEveryRowLeavesAnEmptyCache()
    {
        const int inputSize = 4;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize))
            {
                writer.Append(MakeImage(0), [0]);
                writer.Append(MakeImage(1), [1]);
            }

            var result = PreprocessedDatasetCacheCompactor.DropRows(directory, new HashSet<int> { 0, 1 });

            Assert.Equal(2, result.OriginalCount);
            Assert.Equal(0, result.KeptCount);
            Assert.Empty(result.OldToNewIndex);

            using var reader = new PreprocessedDatasetCacheReader(directory);
            Assert.Equal(0, reader.ImageCount);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void DropRows_ResumesAfterCancellationInsteadOfRestarting()
    {
        const int inputSize = 4;
        const int rowCount = 4001; // crosses two internal checkpoint boundaries (2000, 4000)
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var tagRows = Enumerable.Range(0, rowCount).Select(i => (IReadOnlyList<int>)new List<int> { i }).ToList();
            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize))
            {
                for (var i = 0; i < rowCount; i++)
                    writer.Append(MakeImage(i, width: 1, height: 1), tagRows[i]);
            }

            // Drop every 7th row -- an arbitrary scattered pattern, not just a prefix/suffix.
            var rowsToDrop = Enumerable.Range(0, rowCount).Where(i => i % 7 == 0).ToHashSet();

            using (var cts = new CancellationTokenSource())
            {
                var lastReported = 0;
                Assert.Throws<OperationCanceledException>(() =>
                    PreprocessedDatasetCacheCompactor.DropRows(directory, rowsToDrop, (processed, _) =>
                    {
                        lastReported = processed;
                        if (processed == 2000)
                            cts.Cancel();
                    }, cts.Token));
                Assert.Equal(2000, lastReported);
            }

            // The original (uncompacted) cache must be untouched by the interrupted attempt.
            using (var stillOriginal = new PreprocessedDatasetCacheReader(directory))
                Assert.Equal(rowCount, stillOriginal.ImageCount);
            Assert.True(Directory.Exists(Path.Combine(directory, ".compact-tmp")));

            var result = PreprocessedDatasetCacheCompactor.DropRows(directory, rowsToDrop);

            var expectedKeptOldIndices = Enumerable.Range(0, rowCount).Where(i => !rowsToDrop.Contains(i)).ToList();
            Assert.Equal(rowCount, result.OriginalCount);
            Assert.Equal(expectedKeptOldIndices.Count, result.KeptCount);

            using var reader = new PreprocessedDatasetCacheReader(directory);
            Assert.Equal(expectedKeptOldIndices.Count, reader.ImageCount);
            for (var newIndex = 0; newIndex < expectedKeptOldIndices.Count; newIndex++)
            {
                var oldIndex = expectedKeptOldIndices[newIndex];
                Assert.Equal(newIndex, result.OldToNewIndex[oldIndex]);
                Assert.Equal(tagRows[oldIndex], reader.ImageTagRows[newIndex]);
                Assert.Equal(ImagePreprocessing.Reconstruct(MakeImage(oldIndex, width: 1, height: 1), inputSize).Pixels, reader.ReadImage(newIndex).Pixels);
            }

            Assert.False(Directory.Exists(Path.Combine(directory, ".compact-tmp")));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void DropRows_EmptyDropSetThrows()
    {
        const int inputSize = 4;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize))
                writer.Append(MakeImage(0), [0]);

            Assert.Throws<ArgumentException>(() => PreprocessedDatasetCacheCompactor.DropRows(directory, new HashSet<int>()));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void DetectValidPrefixLength_ReturnsFullCountWhenEveryRowIsReal()
    {
        const int inputSize = 4;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize))
            {
                for (var i = 0; i < 5; i++)
                    writer.Append(MakeImage(i), [i]);
            }

            Assert.Equal(5, PreprocessedDatasetCacheCompactor.DetectValidPrefixLength(directory));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void DetectValidPrefixLength_StopsAtFirstZeroContentRecord()
    {
        const int inputSize = 4;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize))
            {
                for (var i = 0; i < 5; i++)
                    writer.Append(MakeImage(i), [i]);
            }

            // Simulate the write-side failure this method exists to catch: everything
            // from some point on silently written as an empty 0x0 record, despite the
            // header still claiming the full row count.
            var pixelPath = Path.Combine(directory, "images.bin");
            using (var stream = new FileStream(pixelPath, FileMode.Open, FileAccess.ReadWrite))
            {
                stream.Seek(-MakeImage(0).Pixels.Length - 16, SeekOrigin.End); // start of the last record's box header
                using var writer = new BinaryWriter(stream);
                writer.Write(0); // x
                writer.Write(0); // y
                writer.Write(0); // width
                writer.Write(0); // height
            }

            Assert.Equal(4, PreprocessedDatasetCacheCompactor.DetectValidPrefixLength(directory));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void CopyPrefix_CopiesLeadingRowsIntoAFreshDirectoryLeavingSourceUntouched()
    {
        const int inputSize = 8;
        var sourceDirectory = Directory.CreateTempSubdirectory().FullName;
        var destinationDirectory = Path.Combine(Directory.CreateTempSubdirectory().FullName, "dest");
        try
        {
            var images = Enumerable.Range(0, 5).Select(i => MakeImage(i)).ToList();
            var tagRows = Enumerable.Range(0, 5).Select(i => (IReadOnlyList<int>)new List<int> { i, i + 100 }).ToList();
            using (var writer = new PreprocessedDatasetCacheWriter(sourceDirectory, inputSize))
            {
                for (var i = 0; i < images.Count; i++)
                    writer.Append(images[i], tagRows[i]);
            }

            var progressCalls = new List<(int, int)>();
            PreprocessedDatasetCacheCompactor.CopyPrefix(sourceDirectory, destinationDirectory, 3, (done, total) => progressCalls.Add((done, total)));

            using (var source = new PreprocessedDatasetCacheReader(sourceDirectory))
                Assert.Equal(5, source.ImageCount); // untouched

            using var reader = new PreprocessedDatasetCacheReader(destinationDirectory);
            Assert.Equal(3, reader.ImageCount);
            for (var i = 0; i < 3; i++)
            {
                Assert.Equal(ImagePreprocessing.Reconstruct(images[i], inputSize).Pixels, reader.ReadImage(i).Pixels);
                Assert.Equal(tagRows[i], reader.ImageTagRows[i]);
            }
            Assert.Equal((3, 3), progressCalls[^1]);

            // Re-running once already complete is a cheap no-op, not a re-copy.
            PreprocessedDatasetCacheCompactor.CopyPrefix(sourceDirectory, destinationDirectory, 3);
            using var reader2 = new PreprocessedDatasetCacheReader(destinationDirectory);
            Assert.Equal(3, reader2.ImageCount);
        }
        finally
        {
            Directory.Delete(sourceDirectory, recursive: true);
            if (Directory.Exists(destinationDirectory))
                Directory.Delete(destinationDirectory, recursive: true);
        }
    }

    /// <summary>
    /// Regression test for a real near-miss: a caller (dataset recovery) appends MORE
    /// rows onto CopyPrefix's own destination afterward (re-downloading everything past
    /// the valid prefix). Re-running CopyPrefix at that point -- e.g. because the whole
    /// recovery process restarted -- must recognize the destination as already done and
    /// leave it alone, not treat "more rows than requested" as "incomplete" and wipe
    /// out however much of that far more expensive follow-on work had already landed.
    /// </summary>
    [Fact]
    public void CopyPrefix_DestinationGrownPastRowCountByACallerIsLeftAlone()
    {
        const int inputSize = 8;
        var sourceDirectory = Directory.CreateTempSubdirectory().FullName;
        var destinationDirectory = Path.Combine(Directory.CreateTempSubdirectory().FullName, "dest");
        try
        {
            var images = Enumerable.Range(0, 5).Select(i => MakeImage(i)).ToList();
            using (var writer = new PreprocessedDatasetCacheWriter(sourceDirectory, inputSize))
            {
                for (var i = 0; i < images.Count; i++)
                    writer.Append(images[i], [i]);
            }

            PreprocessedDatasetCacheCompactor.CopyPrefix(sourceDirectory, destinationDirectory, 3);

            // Simulate a caller (the download-fill phase) appending further rows onto
            // the same destination after CopyPrefix finished.
            using (var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(destinationDirectory, inputSize))
                writer.Append(MakeImage(99), [99]);

            PreprocessedDatasetCacheCompactor.CopyPrefix(sourceDirectory, destinationDirectory, 3);

            using var reader = new PreprocessedDatasetCacheReader(destinationDirectory);
            Assert.Equal(4, reader.ImageCount); // the appended 4th row must survive, not get wiped
        }
        finally
        {
            Directory.Delete(sourceDirectory, recursive: true);
            if (Directory.Exists(destinationDirectory))
                Directory.Delete(destinationDirectory, recursive: true);
        }
    }
}
