using System.Net;

namespace UnbooruTagger.Crawler;

/// <summary>
/// Shared retry-with-backoff policy, used by both <see cref="BooruHttpClientBase"/>
/// (the JSON listing APIs) and <see cref="DatasetCrawler"/>'s raw image downloads.
/// Retries on a 429 response (honoring <c>Retry-After</c> when the server sends it)
/// AND on network-transport failures — connection refused/reset, DNS, TLS, a request
/// timing out — since an unattended multi-hour crawl will eventually hit a transient
/// blip and a single one used to take the whole run down
/// (<see cref="System.Net.Http.HttpRequestException"/> propagating uncaught out of
/// <c>sendAsync</c>). Otherwise backs off exponentially starting at
/// <see cref="InitialBackoff"/>, capped at <see cref="MaxBackoff"/>, giving up on this
/// one request (not the whole run) after <see cref="MaxRetries"/> attempts.
/// </summary>
public static class TransientHttpRetry
{
    public const int MaxRetries = 8;
    public static readonly TimeSpan InitialBackoff = TimeSpan.FromSeconds(5);
    public static readonly TimeSpan MaxBackoff = TimeSpan.FromMinutes(2);

    /// <summary>
    /// Runs <paramref name="sendAsync"/> (one full attempt: rate-limiter wait + the
    /// actual request) up to <see cref="MaxRetries"/> times, retrying on a 429 response
    /// or a transient network exception. The caller owns rate-limiting and
    /// completion-option choices inside <paramref name="sendAsync"/> since those differ
    /// between a JSON listing call and a raw image download. <paramref name="delay"/> is
    /// injectable (defaults to <see cref="Task.Delay(TimeSpan, CancellationToken)"/>), the
    /// same pattern <see cref="FixedIntervalRateLimiter"/> uses, so retry/backoff
    /// behavior can be verified in tests without a real wait.
    /// </summary>
    public static async Task<HttpResponseMessage> SendWithRetryAsync(
        Func<Task<HttpResponseMessage>> sendAsync,
        Uri uri,
        CancellationToken cancellationToken,
        Func<TimeSpan, CancellationToken, Task>? delay = null)
    {
        delay ??= Task.Delay;

        for (var attempt = 1; ; attempt++)
        {
            HttpResponseMessage response;
            try
            {
                response = await sendAsync().ConfigureAwait(false);
            }
            catch (Exception ex) when (attempt < MaxRetries && IsTransient(ex, cancellationToken))
            {
                await delay(BackoffFor(attempt), cancellationToken).ConfigureAwait(false);
                continue;
            }

            if (response.StatusCode != HttpStatusCode.TooManyRequests)
                return response;

            using (response)
            {
                if (attempt >= MaxRetries)
                    throw new HttpRequestException($"Rate-limited (429) after {MaxRetries} attempts: {uri}");

                var retryAfter = response.Headers.RetryAfter?.Delta ?? BackoffFor(attempt);
                await delay(retryAfter, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    /// <summary>
    /// True for network-transport failures worth retrying. Excludes cancellation
    /// requested by <paramref name="cancellationToken"/> itself — that must propagate
    /// immediately rather than being swallowed into a retry loop.
    /// </summary>
    private static bool IsTransient(Exception ex, CancellationToken cancellationToken) =>
        ex switch
        {
            HttpRequestException => true,
            IOException => true,
            TaskCanceledException => !cancellationToken.IsCancellationRequested,
            _ => false,
        };

    private static TimeSpan BackoffFor(int attempt)
    {
        var seconds = InitialBackoff.TotalSeconds * Math.Pow(2, attempt - 1);
        return TimeSpan.FromSeconds(Math.Min(seconds, MaxBackoff.TotalSeconds));
    }
}
