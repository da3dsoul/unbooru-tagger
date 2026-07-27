using System.Reflection;
using Microsoft.Data.Sqlite;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Tests.Core;

public class CacheResumeSmokeTests
{
    private static EncodedImage Image(int inputSize, byte offset) =>
        new(
            Enumerable.Range(0, 3 * inputSize * inputSize).Select(i => (byte)(i + offset)).ToArray(),
            new LetterboxBox(0, 0, inputSize, inputSize));

    /// <summary>
    /// Force-closes both of <paramref name="writer"/>'s underlying stores WITHOUT going
    /// through <see cref="PreprocessedDatasetCacheWriter.Dispose"/>/<see cref="PreprocessedDatasetCacheWriter.Flush"/>
    /// — simulating a crash mid-page. The pixel side is a plain <c>FileStream</c>
    /// (unchanged since before the tag-row SQLite migration). The tag-row side is a
    /// <c>TagRowStore</c> wrapping a <c>SqliteConnection</c> with a pending, uncommitted
    /// transaction — closing that connection directly (bypassing <c>TagRowStore.Flush</c>/
    /// <c>Dispose</c>, which would commit) rolls the pending transaction back, the same
    /// crash-safety guarantee SQLite already gives for free (see <c>TagRowStore</c>'s own
    /// doc comment). Reached entirely through untyped reflection (<c>object</c>/
    /// <c>GetType()</c>, never <c>typeof(TagRowStore)</c>) since that type is internal.
    /// </summary>
    private static void SimulateCrash(PreprocessedDatasetCacheWriter writer)
    {
        var pixelField = typeof(PreprocessedDatasetCacheWriter).GetField("_pixelStream", BindingFlags.NonPublic | BindingFlags.Instance)!;
        ((FileStream)pixelField.GetValue(writer)!).Close();

        var tagRowStoreField = typeof(PreprocessedDatasetCacheWriter).GetField("_tagRowStore", BindingFlags.NonPublic | BindingFlags.Instance)!;
        var tagRowStore = tagRowStoreField.GetValue(writer)!;
        var connectionField = tagRowStore.GetType().GetField("_connection", BindingFlags.NonPublic | BindingFlags.Instance)!;
        ((SqliteConnection)connectionField.GetValue(tagRowStore)!).Close();
    }

    [Fact]
    public void OpenOrCreate_DropsDanglingPageAndResumesCleanly()
    {
        const int inputSize = 2;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var image0 = Image(inputSize, 0);
            var image1 = Image(inputSize, 100);
            var image2 = Image(inputSize, 200);

            var writer1 = PreprocessedDatasetCacheWriter.OpenOrCreate(directory, inputSize);
            writer1.Append(image0, [0, 2]);
            writer1.Flush(); // "page 1" confirmed committed

            writer1.Append(image1, [9]); // "page 2" started, never flushed -- simulates a mid-page crash
            SimulateCrash(writer1);

            using (var resumed = PreprocessedDatasetCacheWriter.OpenOrCreate(directory, inputSize))
            {
                Assert.Equal(1, resumed.ImageCount); // only the flushed page survived
                resumed.Append(image2, [5]);
            }

            using var reader = new PreprocessedDatasetCacheReader(directory);
            Assert.Equal(2, reader.ImageCount);
            Assert.Equal(new[] { 0, 2 }, reader.ImageTagRows[0]);
            Assert.Equal(new[] { 5 }, reader.ImageTagRows[1]);
            Assert.Equal(ImagePreprocessing.Reconstruct(image0, inputSize).Pixels, reader.ReadImage(0).Pixels);
            Assert.Equal(ImagePreprocessing.Reconstruct(image2, inputSize).Pixels, reader.ReadImage(1).Pixels);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void OpenOrCreate_DropsDanglingPageViaFastPath_WhenPriorResumeIndexIsValid()
    {
        // Same crash scenario as OpenOrCreate_DropsDanglingPageAndResumesCleanly, but
        // this time asserting HOW the pixel side recovers: since the last successful
        // Flush already wrote a resume index for page 1, dropping page 2's dangling
        // pixel bytes should be a direct byte-offset truncate, not a re-walk. There's no
        // equivalent "slow path" left to assert against on the tag-row side anymore —
        // TagRowStore's uncommitted transaction rolling back on close (see
        // SimulateCrash) is what drops page 2's tag row, and that's unconditionally
        // O(1) via SQLite's own transaction log, never a manual scan.
        const int inputSize = 2;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var image0 = Image(inputSize, 0);
            var image1 = Image(inputSize, 100);
            var image2 = Image(inputSize, 200);

            var writer1 = PreprocessedDatasetCacheWriter.OpenOrCreate(directory, inputSize);
            writer1.Append(image0, [0, 2]);
            writer1.Flush(); // "page 1" confirmed committed -- writes a valid resume index

            writer1.Append(image1, [9]); // "page 2" started, never flushed -- simulates a mid-page crash
            SimulateCrash(writer1);

            var walkedPixelIndex = false;
            using (var resumed = PreprocessedDatasetCacheWriter.OpenOrCreate(directory, inputSize,
                       onResumeProgress: (subPhase, _, _) =>
                       {
                           if (subPhase == "resuming pixel index")
                               walkedPixelIndex = true;
                       }))
            {
                Assert.Equal(1, resumed.ImageCount); // only the flushed page survived
                resumed.Append(image2, [5]);
            }

            Assert.False(walkedPixelIndex);

            using var reader = new PreprocessedDatasetCacheReader(directory);
            Assert.Equal(2, reader.ImageCount);
            Assert.Equal(new[] { 0, 2 }, reader.ImageTagRows[0]);
            Assert.Equal(new[] { 5 }, reader.ImageTagRows[1]);
            Assert.Equal(ImagePreprocessing.Reconstruct(image0, inputSize).Pixels, reader.ReadImage(0).Pixels);
            Assert.Equal(ImagePreprocessing.Reconstruct(image2, inputSize).Pixels, reader.ReadImage(1).Pixels);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void OpenOrCreate_ResumesViaIndex_WithoutWalkingWhenIndexIsValid()
    {
        const int inputSize = 2;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using (var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(directory, inputSize))
            {
                writer.Append(Image(inputSize, 0), [0]);
                writer.Append(Image(inputSize, 50), [1]);
                writer.Flush(); // writes a valid images.bin.resume for ImageCount == 2
            }

            Assert.True(File.Exists(Path.Combine(directory, "images.bin.resume")));

            var walkedPixelIndex = false;
            using var resumed = PreprocessedDatasetCacheWriter.OpenOrCreate(directory, inputSize,
                onResumeProgress: (subPhase, _, _) =>
                {
                    if (subPhase == "resuming pixel index")
                        walkedPixelIndex = true;
                });

            Assert.False(walkedPixelIndex); // a valid index means no box-header walk was needed
            Assert.Equal(2, resumed.ImageCount);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void OpenOrCreate_FallsBackAndHealsResumeIndex_WhenMissing()
    {
        const int inputSize = 2;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using (var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(directory, inputSize))
            {
                writer.Append(Image(inputSize, 0), [0]);
                writer.Flush();
            }

            var resumeIndexPath = Path.Combine(directory, "images.bin.resume");
            Assert.True(File.Exists(resumeIndexPath));
            File.Delete(resumeIndexPath);

            var walkedPixelIndex = false;
            using (var resumed = PreprocessedDatasetCacheWriter.OpenOrCreate(directory, inputSize,
                       onResumeProgress: (subPhase, _, _) =>
                       {
                           if (subPhase == "resuming pixel index")
                               walkedPixelIndex = true;
                       }))
            {
                Assert.True(walkedPixelIndex); // no index to trust -- had to fall back to the walk
                Assert.Equal(1, resumed.ImageCount);
                resumed.Append(Image(inputSize, 100), [2]);
            }

            // Healed by the fallback above: the next open shouldn't need to walk either.
            Assert.True(File.Exists(resumeIndexPath));
            var healedWalk = false;
            using var reopened = PreprocessedDatasetCacheWriter.OpenOrCreate(directory, inputSize,
                onResumeProgress: (subPhase, _, _) =>
                {
                    if (subPhase == "resuming pixel index")
                        healedWalk = true;
                });

            Assert.False(healedWalk);
            Assert.Equal(2, reopened.ImageCount);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void OpenOrCreate_FallsBackAndHealsResumeIndex_WhenStale()
    {
        const int inputSize = 2;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using (var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(directory, inputSize))
            {
                writer.Append(Image(inputSize, 0), [0]);
                writer.Flush();
            }

            // Overwrite the index claiming an ImageCount that doesn't match the pixel
            // file's real header (1) -- simulates it being stale/left over.
            var resumeIndexPath = Path.Combine(directory, "images.bin.resume");
            using (var stream = new FileStream(resumeIndexPath, FileMode.Create, FileAccess.Write))
            using (var indexWriter = new BinaryWriter(stream))
            {
                indexWriter.Write(0x52534D31); // "RSM1" magic
                indexWriter.Write(999); // wrong -- real header says 1
                indexWriter.Write(0L);
            }

            var walkedPixelIndex = false;
            using var resumed = PreprocessedDatasetCacheWriter.OpenOrCreate(directory, inputSize,
                onResumeProgress: (subPhase, _, _) =>
                {
                    if (subPhase == "resuming pixel index")
                        walkedPixelIndex = true;
                });

            Assert.True(walkedPixelIndex); // mismatched count rejected, fell back to the walk
            Assert.Equal(1, resumed.ImageCount);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
