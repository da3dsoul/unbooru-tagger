using System.Net;
using System.Net.Http.Headers;
using System.Net.Sockets;
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

    [Fact]
    public async Task SendWithRetryAsync_RetriesOnConnectionFailure_ThenSucceeds()
    {
        // Regression test: a connection-refused/reset used to propagate straight out of
        // sendAsync() uncaught, crashing an unattended multi-hour crawl on the first blip.
        var attempts = 0;
        var delays = new List<TimeSpan>();
        var response = await TransientHttpRetry.SendWithRetryAsync(
            () =>
            {
                attempts++;
                if (attempts < 3)
                    throw new HttpRequestException("Connection refused", new SocketException((int)SocketError.ConnectionRefused));
                return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK));
            },
            new Uri("https://gelbooru.com/index.php"),
            CancellationToken.None,
            delay: (span, _) => { delays.Add(span); return Task.CompletedTask; });

        Assert.Equal(3, attempts);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal([TransientHttpRetry.InitialBackoff, TransientHttpRetry.InitialBackoff * 2], delays);
    }

    [Fact]
    public async Task SendWithRetryAsync_ThrowsAfterMaxRetries_OnPersistentConnectionFailure()
    {
        var attempts = 0;

        var ex = await Assert.ThrowsAsync<HttpRequestException>(() =>
            TransientHttpRetry.SendWithRetryAsync(
                () => { attempts++; throw new HttpRequestException("Connection refused"); },
                new Uri("https://gelbooru.com/index.php"),
                CancellationToken.None,
                delay: (_, _) => Task.CompletedTask));

        Assert.Equal(TransientHttpRetry.MaxRetries, attempts);
        Assert.Contains("Connection refused", ex.Message);
    }

    [Fact]
    public async Task SendWithRetryAsync_DoesNotRetry_WhenCancellationTokenItselfIsCancelled()
    {
        // A user-driven cancellation (Ctrl+C, shutdown) must propagate immediately, not
        // get swallowed into a retry loop the way an ordinary transient failure does.
        using var cts = new CancellationTokenSource();
        var attempts = 0;
        var delayCalled = false;

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            TransientHttpRetry.SendWithRetryAsync(
                () =>
                {
                    attempts++;
                    cts.Cancel();
                    throw new TaskCanceledException("Request timed out");
                },
                new Uri("https://example.com"),
                cts.Token,
                delay: (_, _) => { delayCalled = true; return Task.CompletedTask; }));

        Assert.Equal(1, attempts);
        Assert.False(delayCalled);
    }
}
