using UnbooruTagger.Core.Dataset;

namespace UnbooruTagger.Tests.Core;

public class PreprocessedDatasetCacheTests
{
    [Fact]
    public void WriteAndRead_RoundTripsPixelsAndTagRows()
    {
        const int inputSize = 2;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var image0 = Enumerable.Range(0, 3 * inputSize * inputSize).Select(i => (float)i).ToArray();
            var image1 = Enumerable.Range(0, 3 * inputSize * inputSize).Select(i => (float)(i + 100)).ToArray();

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
            Assert.Equal(image0, reader.ReadImage(0));
            Assert.Equal(image1, reader.ReadImage(1));
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
            Assert.Throws<ArgumentException>(() => writer.Append([1f, 2f], []));
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
            var image = Enumerable.Range(0, 3 * inputSize * inputSize).Select(i => (float)i).ToArray();

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
    public void MergeTagRows_NoOp_WhenNothingToMerge()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var writer = new PreprocessedDatasetCacheWriter(directory, inputSize: 2);
            writer.Append(Enumerable.Range(0, 12).Select(i => (float)i).ToArray(), [0]);
            writer.Flush();

            // Must not touch (or close/reopen) the label file when there's nothing to merge.
            writer.MergeTagRows(new Dictionary<int, IReadOnlyList<int>>());
            writer.Append(Enumerable.Range(0, 12).Select(i => (float)i).ToArray(), [1]);

            Assert.Equal(2, writer.ImageCount);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
