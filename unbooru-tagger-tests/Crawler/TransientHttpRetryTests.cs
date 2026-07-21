using System.Net;
using System.Net.Http.Headers;
using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

public class TransientHttpRetryTests
{
    private static HttpResponseMessage TooManyRequests(int retryAfterSeconds = 0)
    {
        var response = new HttpResponseMessage(HttpStatusCode.TooManyRequests);
        response.Headers.RetryAfter = new RetryConditionHeaderValue(TimeSpan.FromSeconds(retryAfterSeconds));
        return response;
    }

    [Fact]
    public async Task SendWithRetryAsync_ReturnsImmediately_OnFirstSuccess()
    {
        var attempts = 0;
        var response = await TransientHttpRetry.SendWithRetryAsync(
            () => { attempts++; return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)); },
            new Uri("https://example.com"),
            CancellationToken.None);

        Assert.Equal(1, attempts);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task SendWithRetryAsync_RetriesOn429_ThenSucceeds()
    {
        var attempts = 0;
        var response = await TransientHttpRetry.SendWithRetryAsync(
            () =>
            {
                attempts++;
                return Task.FromResult(attempts < 3 ? TooManyRequests() : new HttpResponseMessage(HttpStatusCode.OK));
            },
            new Uri("https://example.com"),
            CancellationToken.None);

        Assert.Equal(3, attempts);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task SendWithRetryAsync_ThrowsAfterMaxRetries_InsteadOfCrashingCaller()
    {
        var attempts = 0;

        var ex = await Assert.ThrowsAsync<HttpRequestException>(() =>
            TransientHttpRetry.SendWithRetryAsync(
                () => { attempts++; return Task.FromResult(TooManyRequests()); },
                new Uri("https://example.com/image.jpg"),
                CancellationToken.None));

        Assert.Equal(TransientHttpRetry.MaxRetries, attempts);
        Assert.Contains("429", ex.Message);
    }

    [Fact]
    public async Task SendWithRetryAsync_DoesNotRetry_OnNonRateLimitStatus()
    {
        var attempts = 0;
        var response = await TransientHttpRetry.SendWithRetryAsync(
            () => { attempts++; return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound)); },
            new Uri("https://example.com"),
            CancellationToken.None);

        Assert.Equal(1, attempts);
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }
}
