using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

public class FixedIntervalRateLimiterTests
{
    [Fact]
    public async Task WaitAsync_DoesNotDelay_FirstCall()
    {
        var now = DateTimeOffset.UtcNow;
        var delays = new List<TimeSpan>();
        var limiter = new FixedIntervalRateLimiter(
            requestsPerSecond: 2, // 500ms interval
            now: () => now,
            delay: (span, _) => { delays.Add(span); return Task.CompletedTask; });

        await limiter.WaitAsync();

        Assert.Empty(delays);
    }

    [Fact]
    public async Task WaitAsync_DelaysSecondCall_WhenWithinInterval()
    {
        var now = DateTimeOffset.UtcNow;
        var delays = new List<TimeSpan>();
        var limiter = new FixedIntervalRateLimiter(
            requestsPerSecond: 2, // 500ms interval
            now: () => now,
            delay: (span, _) => { delays.Add(span); return Task.CompletedTask; });

        await limiter.WaitAsync();
        await limiter.WaitAsync(); // "now" hasn't advanced -> must wait a full interval

        Assert.Single(delays);
        Assert.Equal(TimeSpan.FromMilliseconds(500), delays[0]);
    }

    [Fact]
    public async Task WaitAsync_DoesNotDelay_WhenEnoughRealTimeAlreadyPassed()
    {
        var now = DateTimeOffset.UtcNow;
        var delays = new List<TimeSpan>();
        var limiter = new FixedIntervalRateLimiter(
            requestsPerSecond: 10, // 100ms interval
            now: () => now,
            delay: (span, _) => { delays.Add(span); return Task.CompletedTask; });

        await limiter.WaitAsync();
        now += TimeSpan.FromSeconds(1); // plenty of time passed since the first call
        await limiter.WaitAsync();

        Assert.Empty(delays);
    }

    [Fact]
    public async Task WaitAsync_AccountsForPartialElapsedTime()
    {
        var now = DateTimeOffset.UtcNow;
        var delays = new List<TimeSpan>();
        var limiter = new FixedIntervalRateLimiter(
            requestsPerSecond: 5, // 200ms interval
            now: () => now,
            delay: (span, _) => { delays.Add(span); return Task.CompletedTask; });

        await limiter.WaitAsync();
        now += TimeSpan.FromMilliseconds(50); // only 50ms of the 200ms interval elapsed
        await limiter.WaitAsync();

        Assert.Single(delays);
        Assert.Equal(TimeSpan.FromMilliseconds(150), delays[0]);
    }
}
