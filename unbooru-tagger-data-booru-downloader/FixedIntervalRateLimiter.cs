namespace UnbooruTagger.Crawler;

/// <summary>
/// Enforces a minimum interval (1/requestsPerSecond) between successive calls to
/// <see cref="WaitAsync"/>, suspending the calling <see cref="Task"/> rather than
/// blocking a thread — unlike the sibling <c>unbooru</c> repo's blocking,
/// <c>Thread.Sleep</c>-based <c>SimpleRateLimiter</c>, which isn't safe to await from
/// several concurrent site workers on the thread pool. <paramref name="now"/> and
/// <paramref name="delay"/> are injectable so timing behavior can be verified in tests
/// without real waits.
/// </summary>
public sealed class FixedIntervalRateLimiter : IRateLimiter
{
    private readonly TimeSpan _interval;
    private readonly Func<DateTimeOffset> _now;
    private readonly Func<TimeSpan, CancellationToken, Task> _delay;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private DateTimeOffset? _nextAllowed;

    public FixedIntervalRateLimiter(
        double requestsPerSecond,
        Func<DateTimeOffset>? now = null,
        Func<TimeSpan, CancellationToken, Task>? delay = null)
    {
        if (requestsPerSecond <= 0)
            throw new ArgumentOutOfRangeException(nameof(requestsPerSecond), "Must be positive.");

        _interval = TimeSpan.FromSeconds(1.0 / requestsPerSecond);
        _now = now ?? (() => DateTimeOffset.UtcNow);
        _delay = delay ?? Task.Delay;
    }

    public async Task WaitAsync(CancellationToken cancellationToken = default)
    {
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var now = _now();
            var scheduledStart = _nextAllowed is { } next && next > now ? next : now;

            if (scheduledStart > now)
                await _delay(scheduledStart - now, cancellationToken).ConfigureAwait(false);

            _nextAllowed = scheduledStart + _interval;
        }
        finally
        {
            _gate.Release();
        }
    }
}
