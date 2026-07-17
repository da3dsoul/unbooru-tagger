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
}
