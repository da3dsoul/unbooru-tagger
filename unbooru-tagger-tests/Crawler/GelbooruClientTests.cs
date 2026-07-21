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

/// <summary>Like <see cref="FakeHttpMessageHandler"/> but records every request URI it saw, for asserting on query parameters.</summary>
internal sealed class RecordingHttpMessageHandler(string responseBody) : HttpMessageHandler
{
    public List<Uri> RequestUris { get; } = [];

    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        RequestUris.Add(request.RequestUri!);
        return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(responseBody) });
    }
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
    public async Task ListTagsByCountDescendingAsync_RequestsCountDescendingOrder()
    {
        const string emptyTagResponse = """{ "tag": [] }""";
        var handler = new RecordingHttpMessageHandler(emptyTagResponse);
        var httpClient = new HttpClient(handler);
        var client = new GelbooruClient(httpClient, new ImmediateRateLimiter());

        await foreach (var _ in client.ListTagsByCountDescendingAsync())
        {
        }

        Assert.NotEmpty(handler.RequestUris);
        var query = handler.RequestUris[0].Query;
        // Gelbooru's dapi splits sort field (orderby) from sort direction (order) —
        // orderby=count alone silently falls back to an unspecified order.
        Assert.Contains("orderby=count", query);
        Assert.Contains("order=DESC", query);
    }

    [Fact]
    public async Task ListPostsAsync_FirstPage_RequestsSortByIdDescendingWithNoIdFilter()
    {
        var handler = new RecordingHttpMessageHandler(SamplePostJson);
        var httpClient = new HttpClient(handler);
        var client = new GelbooruClient(httpClient, new ImmediateRateLimiter());

        await client.ListPostsAsync("1girl", cursor: null);

        var query = Uri.UnescapeDataString(handler.RequestUris[0].Query);
        Assert.Contains("sort:id:desc", query);
        Assert.DoesNotContain("id:<", query);
        // pid was a raw page-number offset — unsafe on a constantly-growing site, since
        // posts added between two runs shift what "page N" means and can silently skip
        // or duplicate posts across a resumed crawl. Anchoring on id:< instead means
        // this request must not use it at all.
        Assert.DoesNotContain("pid=", query);
    }

    [Fact]
    public async Task ListPostsAsync_WithCursor_FiltersToIdsBelowCursor()
    {
        var handler = new RecordingHttpMessageHandler(SamplePostJson);
        var httpClient = new HttpClient(handler);
        var client = new GelbooruClient(httpClient, new ImmediateRateLimiter());

        await client.ListPostsAsync("1girl", cursor: "999999");

        var query = Uri.UnescapeDataString(handler.RequestUris[0].Query);
        Assert.Contains("id:<999999", query);
    }

    [Fact]
    public async Task ListPostsAsync_NextCursor_IsLastPostIdNotPageOffset()
    {
        var ids = Enumerable.Range(0, 100).Select(i => 1000000 - i).ToList(); // descending, like sort:id:desc
        var postsJson = string.Join(",", ids.Select(id => $$"""
            { "id": {{id}}, "md5": "md5{{id}}", "file_url": "https://gelbooru.com/images/x/{{id}}.jpg",
              "tags": "1girl", "rating": "general", "created_at": "Fri Jan 01 00:00:00 +0000 2021",
              "width": 100, "height": 100 }
            """));
        var fullPageJson = $$"""{ "post": [{{postsJson}}] }""";

        var httpClient = new HttpClient(new FakeHttpMessageHandler(fullPageJson));
        var client = new GelbooruClient(httpClient, new ImmediateRateLimiter());

        var page = await client.ListPostsAsync("1girl", cursor: null);

        Assert.Equal(ids[^1].ToString(), page.NextCursor); // the last (lowest) id in the page, not "page 1"
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
