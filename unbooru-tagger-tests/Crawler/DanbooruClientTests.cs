using System.Net;
using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

/// <summary>Returns <paramref name="firstPageBody"/> for the first request and an empty JSON array for every request after that — needed for any paginated endpoint (<see cref="DanbooruClient.ListActiveTagAliasesAsync"/> keeps paging until it sees an empty page), since <see cref="FakeHttpMessageHandler"/>'s single fixed response never terminates that loop.</summary>
internal sealed class SinglePageThenEmptyHttpMessageHandler(string firstPageBody) : HttpMessageHandler
{
    private int _calls;

    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var body = Interlocked.Increment(ref _calls) == 1 ? firstPageBody : "[]";
        return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(body) });
    }
}

public class DanbooruClientTests
{
    [Fact]
    public async Task ListActiveTagAliasesAsync_ParsesAntecedentAndConsequentNames()
    {
        const string json =
            """
            [
              { "id": 1, "antecedent_name": "head_pat", "consequent_name": "headpat", "status": "active" },
              { "id": 2, "antecedent_name": "mindbreak", "consequent_name": "mind_break", "status": "active" }
            ]
            """;
        var httpClient = new HttpClient(new SinglePageThenEmptyHttpMessageHandler(json));
        var client = new DanbooruClient(httpClient, new ImmediateRateLimiter());

        var aliases = new List<BooruTagAlias>();
        await foreach (var alias in client.ListActiveTagAliasesAsync())
            aliases.Add(alias);

        Assert.Equal(2, aliases.Count);
        Assert.Contains(aliases, a => a.Antecedent == "head_pat" && a.Consequent == "headpat");
        Assert.Contains(aliases, a => a.Antecedent == "mindbreak" && a.Consequent == "mind_break");
    }

    [Fact]
    public async Task ListActiveTagAliasesAsync_RequestsOnlyActiveStatus()
    {
        const string emptyResponse = "[]";
        var handler = new RecordingHttpMessageHandler(emptyResponse);
        var httpClient = new HttpClient(handler);
        var client = new DanbooruClient(httpClient, new ImmediateRateLimiter());

        await foreach (var _ in client.ListActiveTagAliasesAsync())
        {
        }

        Assert.NotEmpty(handler.RequestUris);
        // Uri doesn't percent-encode '[' / ']' in the query — same as this codebase's
        // existing unescaped search[order]=count/search[hide_empty]=true params.
        Assert.Contains("search[status]=active", handler.RequestUris[0].Query);
    }

    [Fact]
    public async Task ListActiveTagAliasesAsync_SkipsEntriesMissingExpectedFields()
    {
        const string json = """[ { "id": 1, "status": "active" } ]""";
        var httpClient = new HttpClient(new SinglePageThenEmptyHttpMessageHandler(json));
        var client = new DanbooruClient(httpClient, new ImmediateRateLimiter());

        var aliases = new List<BooruTagAlias>();
        await foreach (var alias in client.ListActiveTagAliasesAsync())
            aliases.Add(alias);

        Assert.Empty(aliases);
    }

    [Fact]
    public async Task ListActiveTagAliasesAsync_StopsPaging_OnEmptyPage()
    {
        var handler = new RecordingHttpMessageHandler("[]");
        var httpClient = new HttpClient(handler);
        var client = new DanbooruClient(httpClient, new ImmediateRateLimiter());

        var aliases = new List<BooruTagAlias>();
        await foreach (var alias in client.ListActiveTagAliasesAsync())
            aliases.Add(alias);

        Assert.Empty(aliases);
        Assert.Single(handler.RequestUris); // stopped after the first (empty) page, no second request
    }
}
