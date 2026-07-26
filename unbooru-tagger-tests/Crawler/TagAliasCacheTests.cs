using System.Net;
using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

public class TagAliasCacheTests
{
    [Fact]
    public async Task TryLoadAsync_ReturnsNull_WhenNoCacheFileExists()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var result = await TagAliasCache.TryLoadAsync(directory);
            Assert.Null(result);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task FetchAndCacheAsync_ReturnsNull_AndWritesNoFile_WhenNoDanbooruClientConfigured()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var clients = new Dictionary<string, IBooruClient>
            {
                ["gelbooru"] = new FakeSurveyClient("gelbooru", []),
            };

            var result = await TagAliasCache.FetchAndCacheAsync(directory, clients);

            Assert.Null(result);
            Assert.False(File.Exists(Path.Combine(directory, TagAliasCache.FileName)));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task FetchAndCacheAsync_WritesACacheFile_ThatTryLoadAsyncCanReadBack()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            const string json =
                """
                [
                  { "antecedent_name": "head_pat", "consequent_name": "headpat", "status": "active" }
                ]
                """;
            var httpClient = new HttpClient(new SinglePageThenEmptyHttpMessageHandler(json));
            var danbooru = new DanbooruClient(httpClient, new ImmediateRateLimiter());
            var clients = new Dictionary<string, IBooruClient> { ["danbooru"] = danbooru };

            var fetched = await TagAliasCache.FetchAndCacheAsync(directory, clients);

            Assert.NotNull(fetched);
            Assert.Equal("headpat", fetched!["head_pat"]);
            Assert.True(File.Exists(Path.Combine(directory, TagAliasCache.FileName)));

            var reloaded = await TagAliasCache.TryLoadAsync(directory);
            Assert.NotNull(reloaded);
            Assert.Equal("headpat", reloaded!["head_pat"]);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task FetchAndCacheAsync_PropagatesTheFailure_InsteadOfSwallowingItOrCachingAnEmptyResult()
    {
        // Program.cs's survey-tags/refresh-tags handlers depend on this call actually
        // throwing when Danbooru is unreachable — that's what lets them exit gracefully
        // with a clear message instead of either crashing with a raw stack trace deeper
        // in the call chain, or silently proceeding as if there were no aliases at all.
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var httpClient = new HttpClient(new FakeHttpMessageHandler("Internal Server Error", HttpStatusCode.InternalServerError));
            var danbooru = new DanbooruClient(httpClient, new ImmediateRateLimiter());
            var clients = new Dictionary<string, IBooruClient> { ["danbooru"] = danbooru };

            await Assert.ThrowsAsync<HttpRequestException>(() => TagAliasCache.FetchAndCacheAsync(directory, clients));

            Assert.False(File.Exists(Path.Combine(directory, TagAliasCache.FileName)));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task FetchAndCacheAsync_OverwritesAnExistingCache_WithFreshData()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            await File.WriteAllTextAsync(Path.Combine(directory, TagAliasCache.FileName), """{"stale_antecedent":"stale_consequent"}""");

            const string json = """[ { "antecedent_name": "head_pat", "consequent_name": "headpat", "status": "active" } ]""";
            var httpClient = new HttpClient(new SinglePageThenEmptyHttpMessageHandler(json));
            var danbooru = new DanbooruClient(httpClient, new ImmediateRateLimiter());
            var clients = new Dictionary<string, IBooruClient> { ["danbooru"] = danbooru };

            await TagAliasCache.FetchAndCacheAsync(directory, clients);

            var reloaded = await TagAliasCache.TryLoadAsync(directory);
            Assert.NotNull(reloaded);
            Assert.False(reloaded!.ContainsKey("stale_antecedent")); // stale cache content is gone, not merged
            Assert.Equal("headpat", reloaded["head_pat"]);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
