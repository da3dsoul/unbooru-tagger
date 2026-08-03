using System.Net;
using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

/// <summary>
/// Routes by URL so one <see cref="HttpClient"/> can back both a <see cref="GelbooruClient"/>
/// (HTML alias listing) and a <see cref="DanbooruClient"/> (JSON <c>tag_aliases.json</c>)
/// in the same test — each source returns its first-page body once, then an empty page
/// of its own shape (matching <see cref="SinglePageThenEmptyHttpMessageHandler"/>'s
/// per-endpoint pagination termination) for every call after.
/// </summary>
internal sealed class MultiSourceHttpMessageHandler(string gelbooruHtmlFirstPage, string danbooruJsonFirstPage) : HttpMessageHandler
{
    private int _gelbooruCalls;
    private int _danbooruCalls;

    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var isGelbooru = request.RequestUri!.Query.Contains("page=alias");
        var body = isGelbooru
            ? Interlocked.Increment(ref _gelbooruCalls) == 1 ? gelbooruHtmlFirstPage : "<html></html>"
            : Interlocked.Increment(ref _danbooruCalls) == 1 ? danbooruJsonFirstPage : "[]";
        return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(body) });
    }
}

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
    public async Task FetchAndCacheAsync_FetchesFromGelbooruAlone_WhenNoDanbooruClientConfigured()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            const string html =
                """
                <td><a href="index.php?page=post&amp;s=list&amp;tags=curvy_figure">curvy figure</a> <span class="tag-count">1241</span> <b>&rarr;</b> <a href="index.php?page=post&amp;s=list&amp;tags=curvy">curvy</a> <span class="tag-count">173668</span></td>
                """;
            var httpClient = new HttpClient(new SinglePageThenEmptyHttpMessageHandler(html));
            var gelbooru = new GelbooruClient(httpClient, new ImmediateRateLimiter());
            var clients = new Dictionary<string, IBooruClient> { ["gelbooru"] = gelbooru };

            var fetched = await TagAliasCache.FetchAndCacheAsync(directory, clients);

            Assert.NotNull(fetched);
            Assert.Equal("curvy", fetched!["curvy_figure"]);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task FetchAndCacheAsync_MergesBothSites_DanbooruWinningOnOverlap()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            const string gelbooruHtml =
                """
                <td><a href="index.php?page=post&amp;s=list&amp;tags=curvy_figure">curvy figure</a> <span class="tag-count">1241</span> <b>&rarr;</b> <a href="index.php?page=post&amp;s=list&amp;tags=curvy">curvy</a> <span class="tag-count">173668</span></td>
                <td><a href="index.php?page=post&amp;s=list&amp;tags=head_pat">head pat</a> <span class="tag-count">1</span> <b>&rarr;</b> <a href="index.php?page=post&amp;s=list&amp;tags=gelbooru_target">gelbooru target</a> <span class="tag-count">2</span></td>
                """;
            const string danbooruJson =
                """[ { "antecedent_name": "head_pat", "consequent_name": "headpat", "status": "active" } ]""";

            var httpClient = new HttpClient(new MultiSourceHttpMessageHandler(gelbooruHtml, danbooruJson));
            var gelbooru = new GelbooruClient(httpClient, new ImmediateRateLimiter());
            var danbooru = new DanbooruClient(httpClient, new ImmediateRateLimiter());
            var clients = new Dictionary<string, IBooruClient> { ["gelbooru"] = gelbooru, ["danbooru"] = danbooru };

            var fetched = await TagAliasCache.FetchAndCacheAsync(directory, clients);

            Assert.NotNull(fetched);
            Assert.Equal("curvy", fetched!["curvy_figure"]); // Gelbooru-only antecedent, no conflict
            Assert.Equal("headpat", fetched["head_pat"]); // both sites claim this antecedent — Danbooru wins
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
