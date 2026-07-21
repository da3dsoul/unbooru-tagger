using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

public class CacheConsistencyTests
{
    [Fact]
    public void Validate_Passes_WhenEveryRowIsInBoundsAndUnique()
    {
        var images = new List<(string Md5, int CacheRowIndex, ulong PHash)>
        {
            ("a", 0, 0), ("b", 1, 0), ("c", 2, 0),
        };

        CacheConsistency.Validate(images, writerImageCount: 3, outputDirectory: "/tmp/dataset");
    }

    [Fact]
    public void Validate_Throws_WhenTwoImagesClaimTheSameCacheRow()
    {
        // The exact corruption a cache-file reset while crawl.sqlite is kept produces:
        // a later run's writer restarts counting from a lower ImageCount, reassigning
        // an already-claimed low index to an unrelated new image.
        var images = new List<(string Md5, int CacheRowIndex, ulong PHash)>
        {
            ("original", 0, 0), ("new-but-colliding", 0, 0),
        };

        var ex = Assert.Throws<InvalidDataException>(() =>
            CacheConsistency.Validate(images, writerImageCount: 1, outputDirectory: "/tmp/dataset"));
        Assert.Contains("original", ex.Message);
        Assert.Contains("new-but-colliding", ex.Message);
    }

    [Fact]
    public void Validate_Throws_WhenACacheRowIsOutOfBounds()
    {
        var images = new List<(string Md5, int CacheRowIndex, ulong PHash)> { ("a", 5, 0) };

        var ex = Assert.Throws<InvalidDataException>(() =>
            CacheConsistency.Validate(images, writerImageCount: 3, outputDirectory: "/tmp/dataset"));
        Assert.Contains("5", ex.Message);
    }
}
