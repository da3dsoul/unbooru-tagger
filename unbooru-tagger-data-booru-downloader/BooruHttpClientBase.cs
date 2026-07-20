using System.Net;

namespace UnbooruTagger.Crawler;

/// <summary>
/// Shared rate-limited-GET-with-429-retry plumbing for <see cref="DanbooruClient"/> and
/// <see cref="GelbooruClient"/>. Honors <c>Retry-After</c> when the site sends it,
/// otherwise backs off exponentially starting at <see cref="InitialBackoff"/>, giving up
/// on this one page (not the whole tag/site) after <see cref="MaxRetries"/> attempts.
/// </summary>
public abstract class BooruHttpClientBase(HttpClient http, IRateLimiter rateLimiter)
{
    private const int MaxRetries = 5;
    private static readonly TimeSpan InitialBackoff = TimeSpan.FromSeconds(5);

    protected async Task<string> GetStringAsync(Uri uri, CancellationToken cancellationToken)
    {
        for (var attempt = 1; ; attempt++)
        {
            await rateLimiter.WaitAsync(cancellationToken).ConfigureAwait(false);

            using var response = await http.GetAsync(uri, cancellationToken).ConfigureAwait(false);
            if (response.StatusCode != HttpStatusCode.TooManyRequests)
            {
                response.EnsureSuccessStatusCode();
                return await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            }

            if (attempt >= MaxRetries)
                throw new HttpRequestException($"Rate-limited (429) after {MaxRetries} attempts: {uri}");

            var delay = response.Headers.RetryAfter?.Delta
                        ?? TimeSpan.FromSeconds(InitialBackoff.TotalSeconds * Math.Pow(2, attempt - 1));
            await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
        }
    }
}
