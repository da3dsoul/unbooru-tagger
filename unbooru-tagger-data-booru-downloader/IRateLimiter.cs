namespace UnbooruTagger.Crawler;

/// <summary>
/// A rate limiter a caller awaits before making a request. Deliberately this small
/// (one method) so it's trivial to fake in tests and to swap per-site instances.
/// </summary>
public interface IRateLimiter
{
    /// <summary>Completes once it's this caller's turn to proceed, having waited as long as the limiter's policy requires.</summary>
    Task WaitAsync(CancellationToken cancellationToken = default);
}
