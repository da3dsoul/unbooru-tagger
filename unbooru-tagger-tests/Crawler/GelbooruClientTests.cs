using System.Net;
using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

/// <summary>Returns a fixed response for every request, regardless of URI — enough to unit-test response parsing without a real network call.</summary>
internal sealed class FakeHttpMessageHandler(string responseBody, HttpStatusCode statusCode = HttpStatusCode.OK) : HttpMessageHandler
{
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
        Task.FromResult(new HttpResponseMessage(statusCode) { Content = new StringContent(responseBody) });
}

internal sealed class ImmediateRateLimiter : IRateLimiter
{
    public Task WaitAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
}

public class GelbooruClientTests
{
    private const string SamplePostJson =
        """
        {
          "post": [
            {
              "id": 123,
              "md5": "abc123",
              "file_url": "https://gelbooru.com/images/ab/c1/abc123.jpg",
              "tags": "1girl solo &amp;_(ampersand) blue_sky",
              "rating": "general",
              "created_at": "Fri Jan 01 00:00:00 +0000 2021",
              "width": 1000,
              "height": 1500
            }
          ]
        }
        """;

    [Fact]
    public async Task ListPostsAsync_HtmlDecodesTagNames()
    {
        var httpClient = new HttpClient(new FakeHttpMessageHandler(SamplePostJson));
        var client = new GelbooruClient(httpClient, new ImmediateRateLimiter());

        var page = await client.ListPostsAsync("1girl", cursor: null);

        Assert.Single(page.Posts);
        Assert.Contains("&_(ampersand)", page.Posts[0].Tags); // decoded from "&amp;_(ampersand)"
    }

    [Fact]
    public async Task ListPostsAsync_ParsesAsctimeStyleCreatedAt()
    {
        var httpClient = new HttpClient(new FakeHttpMessageHandler(SamplePostJson));
        var client = new GelbooruClient(httpClient, new ImmediateRateLimiter());

        var page = await client.ListPostsAsync("1girl", cursor: null);

        var expected = new DateTimeOffset(2021, 1, 1, 0, 0, 0, TimeSpan.Zero);
        Assert.Equal(expected, page.Posts[0].CreatedAt);
    }

    [Fact]
    public async Task ListPostsAsync_ReadsRatingFirstCharacterOnly()
    {
        var httpClient = new HttpClient(new FakeHttpMessageHandler(SamplePostJson));
        var client = new GelbooruClient(httpClient, new ImmediateRateLimiter());

        var page = await client.ListPostsAsync("1girl", cursor: null);

        // "general" -> "g", forward/backward compatible with the old single-letter rating scheme.
        Assert.Equal("g", page.Posts[0].Rating);
    }

    [Fact]
    public async Task ListPostsAsync_NextCursorNull_WhenPageShorterThanPageSize()
    {
        var httpClient = new HttpClient(new FakeHttpMessageHandler(SamplePostJson));
        var client = new GelbooruClient(httpClient, new ImmediateRateLimiter());

        var page = await client.ListPostsAsync("1girl", cursor: null);

        Assert.Null(page.NextCursor); // only 1 post returned, well under the 100 page size
    }

    [Fact]
    public async Task ListPostsAsync_SkipsPostsMissingMd5OrFileUrl()
    {
        const string json =
            """
            { "post": [ { "id": 1, "rating": "general", "created_at": "Fri Jan 01 00:00:00 +0000 2021", "tags": "a", "width": 1, "height": 1 } ] }
            """;
        var httpClient = new HttpClient(new FakeHttpMessageHandler(json));
        var client = new GelbooruClient(httpClient, new ImmediateRateLimiter());

        var page = await client.ListPostsAsync("a", cursor: null);

        Assert.Empty(page.Posts);
    }
}
