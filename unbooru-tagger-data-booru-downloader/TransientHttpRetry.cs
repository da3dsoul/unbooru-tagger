using System.Net;

namespace UnbooruTagger.Crawler;

/// <summary>
/// Shared 429-retry-with-backoff policy, used by both <see cref="BooruHttpClientBase"/>
/// (the JSON listing APIs) and <see cref="DatasetCrawler"/>'s raw image downloads. Honors
/// <c>Retry-After</c> when the server sends it, otherwise backs off exponentially
/// starting at <see cref="InitialBackoff"/>, giving up on this one request (not the
/// whole run) after <see cref="MaxRetries"/> attempts.
/// </summary>
public static class TransientHttpRetry
{
    public const int MaxRetries = 5;
    public static readonly TimeSpan InitialBackoff = TimeSpan.FromSeconds(5);

    /// <summary>
    /// Runs <paramref name="sendAsync"/> (one full attempt: rate-limiter wait + the
    /// actual request) up to <see cref="MaxRetries"/> times, retrying only on a 429
    /// response. The caller owns rate-limiting and completion-option choices inside
    /// <paramref name="sendAsync"/> since those differ between a JSON listing call and a
    /// raw image download.
    /// </summary>
    public static async Task<HttpResponseMessage> SendWithRetryAsync(
        Func<Task<HttpResponseMessage>> sendAsync,
        Uri uri,
        CancellationToken cancellationToken)
    {
        for (var attempt = 1; ; attempt++)
        {
            var response = await sendAsync().ConfigureAwait(false);
            if (response.StatusCode != HttpStatusCode.TooManyRequests)
                return response;

            using (response)
            {
                if (attempt >= MaxRetries)
                    throw new HttpRequestException($"Rate-limited (429) after {MaxRetries} attempts: {uri}");

                var delay = response.Headers.RetryAfter?.Delta
                            ?? TimeSpan.FromSeconds(InitialBackoff.TotalSeconds * Math.Pow(2, attempt - 1));
                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
            }
        }
    }
}
