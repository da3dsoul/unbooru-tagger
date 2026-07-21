namespace UnbooruTagger.Crawler;

/// <summary>
/// Shared rate-limited-GET plumbing for <see cref="DanbooruClient"/> and
/// <see cref="GelbooruClient"/>. 429 retry itself lives in <see cref="TransientHttpRetry"/>,
/// shared with <see cref="DatasetCrawler"/>'s raw image downloads — those go through the
/// same per-site <see cref="RateLimiter"/> (exposed via <see cref="IBooruClient"/>) since
/// they hit the same site/CDN and are subject to the same throttling in practice.
/// </summary>
public abstract class BooruHttpClientBase(HttpClient http, IRateLimiter rateLimiter)
{
    /// <summary>The same per-site limiter that gates this client's own JSON listing calls — reused for raw image downloads too (see <see cref="IBooruClient.RateLimiter"/>).</summary>
    public IRateLimiter RateLimiter { get; } = rateLimiter;

    protected async Task<string> GetStringAsync(Uri uri, CancellationToken cancellationToken)
    {
        using var response = await TransientHttpRetry.SendWithRetryAsync(
            async () =>
            {
                await RateLimiter.WaitAsync(cancellationToken).ConfigureAwait(false);
                return await http.GetAsync(uri, cancellationToken).ConfigureAwait(false);
            },
            uri,
            cancellationToken).ConfigureAwait(false);

        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
    }
}
