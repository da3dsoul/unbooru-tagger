using UnbooruTagger.Core.Dataset;

namespace UnbooruTagger.Tests.Core;

public class DatasetManifestTests
{
    [Fact]
    public void SaveAndLoad_RoundTripsEntries()
    {
        var manifest = new DatasetManifest([
            new DatasetImageEntry("images/a.png", ["solo", "1girl"]),
            new DatasetImageEntry("images/b.png", ["nurse_costume"])
        ]);

        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.json");
        try
        {
            manifest.Save(path);
            var loaded = DatasetManifest.Load(path);

            Assert.Equal(2, loaded.Entries.Count);
            Assert.Equal(["solo", "1girl"], loaded.Entries[0].Tags);
            Assert.Equal("images/b.png", loaded.Entries[1].ImagePath);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
