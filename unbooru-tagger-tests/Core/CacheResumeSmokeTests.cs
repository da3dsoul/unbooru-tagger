using System.Reflection;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Tests.Core;

public class CacheResumeSmokeTests
{
    private static EncodedImage Image(int inputSize, byte offset) =>
        new(
            Enumerable.Range(0, 3 * inputSize * inputSize).Select(i => (byte)(i + offset)).ToArray(),
            new LetterboxBox(0, 0, inputSize, inputSize));

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

            // Force-close the raw handles without going through Dispose (which would Flush).
            var pixelField = typeof(PreprocessedDatasetCacheWriter).GetField("_pixelStream", BindingFlags.NonPublic | BindingFlags.Instance)!;
            ((FileStream)pixelField.GetValue(writer1)!).Close();
            var labelField = typeof(PreprocessedDatasetCacheWriter).GetField("_labelWriter", BindingFlags.NonPublic | BindingFlags.Instance)!;
            ((StreamWriter)labelField.GetValue(writer1)!).BaseStream.Close();

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
}
