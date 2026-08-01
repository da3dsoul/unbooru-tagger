using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

public class CrawlCommandRecordTests
{
    private static CrawlCommandRecord SampleRecord() => new(
        Sites: ["danbooru", "gelbooru"],
        MinImages: 500,
        MaxImages: 1000,
        InputSize: 512,
        DanbooruLogin: null,
        DanbooruApiKey: null,
        GelbooruApiKey: "some-api-key",
        GelbooruUserId: "12345",
        RateDanbooru: 4.0,
        RateGelbooru: 2.0,
        NegativeTarget: 1000,
        VocabCompactInterval: 20,
        NegativeCooccurrenceRatio: 0.5,
        NegativeCooccurrenceMinExamples: 15,
        MaxHardNegativeSources: 3);

    [Fact]
    public async Task TryLoadAsync_ReturnsNull_WhenNoRecordFileExists()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var result = await CrawlCommandRecord.TryLoadAsync(directory);
            Assert.Null(result);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task SaveAsync_WritesARecord_ThatTryLoadAsyncCanReadBack()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var record = SampleRecord();

            await CrawlCommandRecord.SaveAsync(directory, record);

            Assert.True(File.Exists(Path.Combine(directory, CrawlCommandRecord.FileName)));

            var reloaded = await CrawlCommandRecord.TryLoadAsync(directory);
            Assert.NotNull(reloaded);
            Assert.Equal(record.Sites, reloaded!.Sites);
            Assert.Equal(record with { Sites = [] }, reloaded with { Sites = [] });
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task SaveAsync_OverwritesAnExistingRecord_WithFreshData()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            await CrawlCommandRecord.SaveAsync(directory, SampleRecord());

            var updated = SampleRecord() with { MaxImages = 2000, GelbooruApiKey = "different-key" };
            await CrawlCommandRecord.SaveAsync(directory, updated);

            var reloaded = await CrawlCommandRecord.TryLoadAsync(directory);
            Assert.NotNull(reloaded);
            Assert.Equal(updated.Sites, reloaded!.Sites);
            Assert.Equal(updated with { Sites = [] }, reloaded with { Sites = [] });
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
