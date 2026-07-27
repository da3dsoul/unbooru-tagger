using System.Net;
using SkiaSharp;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

/// <summary>Returns one fixed post on the first (cursor-less) page, then an exhausted empty page — enough to drive one image through <see cref="DatasetCrawler"/> without a real network call.</summary>
internal sealed class ConcurrencyTestClient(string siteName, BooruPost post) : IBooruClient
{
    public string SiteName => siteName;
    public int PageSize => 100;
    public IRateLimiter RateLimiter { get; } = new ImmediateRateLimiter();

    public IAsyncEnumerable<BooruTagCount> ListTagsByCountDescendingAsync(CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<BooruPostPage> ListPostsAsync(string tagQuery, string? cursor, CancellationToken cancellationToken = default) =>
        Task.FromResult(cursor is null ? new BooruPostPage([post], null) : new BooruPostPage([], null));

    public Task<BooruPost?> GetPostAsync(long postId, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();
}

/// <summary>Returns a fixed multi-post page on the first (cursor-less) call, then an exhausted empty page.</summary>
internal sealed class MultiPostSiteClient(string siteName, IReadOnlyList<BooruPost> posts) : IBooruClient
{
    public string SiteName => siteName;
    public int PageSize => 100;
    public IRateLimiter RateLimiter { get; } = new ImmediateRateLimiter();

    public IAsyncEnumerable<BooruTagCount> ListTagsByCountDescendingAsync(CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<BooruPostPage> ListPostsAsync(string tagQuery, string? cursor, CancellationToken cancellationToken = default) =>
        Task.FromResult(cursor is null ? new BooruPostPage(posts, null) : new BooruPostPage([], null));

    public Task<BooruPost?> GetPostAsync(long postId, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();
}

/// <summary>Answers each requested cursor (null for the first page) from a fixed lookup — for simulating a resumed run that starts partway through a tag's pagination instead of always from page 1, unlike <see cref="MultiPostSiteClient"/>.</summary>
internal sealed class CursorPagedSiteClient(string siteName, IReadOnlyDictionary<string?, BooruPostPage> pagesByCursor) : IBooruClient
{
    public string SiteName => siteName;
    public int PageSize => 100;
    public IRateLimiter RateLimiter { get; } = new ImmediateRateLimiter();

    public IAsyncEnumerable<BooruTagCount> ListTagsByCountDescendingAsync(CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<BooruPostPage> ListPostsAsync(string tagQuery, string? cursor, CancellationToken cancellationToken = default) =>
        Task.FromResult(pagesByCursor.GetValueOrDefault(cursor, new BooruPostPage([], null)));

    public Task<BooruPost?> GetPostAsync(long postId, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();
}

/// <summary>
/// Like <see cref="CursorPagedSiteClient"/> but counts every <see cref="ListPostsAsync"/>
/// call — for proving a resumed run whose quota is already known-satisfied never fetches
/// another page at all, not just that it produces the right data if it did. Takes the
/// first (cursor-null) page separately from <paramref name="pagesByCursor"/> since
/// <c>Dictionary</c> never allows a null key at runtime, even for a nullable-annotated
/// key type — unlike <see cref="CursorPagedSiteClient"/>'s tests, which never needed a
/// null-cursor entry because they only ever simulate resuming PAST page 1.
/// </summary>
internal sealed class CountingCursorPagedSiteClient(string siteName, BooruPostPage firstPage, IReadOnlyDictionary<string, BooruPostPage>? pagesByCursor = null) : IBooruClient
{
    public string SiteName => siteName;
    public int PageSize => 100;
    public IRateLimiter RateLimiter { get; } = new ImmediateRateLimiter();
    public int ListPostsCallCount { get; private set; }

    public IAsyncEnumerable<BooruTagCount> ListTagsByCountDescendingAsync(CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<BooruPostPage> ListPostsAsync(string tagQuery, string? cursor, CancellationToken cancellationToken = default)
    {
        ListPostsCallCount++;
        if (cursor is null)
            return Task.FromResult(firstPage);
        return Task.FromResult(pagesByCursor?.GetValueOrDefault(cursor) ?? new BooruPostPage([], null));
    }

    public Task<BooruPost?> GetPostAsync(long postId, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();
}

/// <summary>Cancels <paramref name="cts"/> on the Nth request, then throws through the request's own token — simulating an exit request noticed mid-download rather than a network failure.</summary>
internal sealed class CancelOnRequestHttpMessageHandler(byte[] imageBytes, int cancelOnRequestNumber, CancellationTokenSource cts) : HttpMessageHandler
{
    private int _requests;

    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        if (Interlocked.Increment(ref _requests) == cancelOnRequestNumber)
        {
            cts.Cancel();
            cancellationToken.ThrowIfCancellationRequested();
        }

        return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(imageBytes) });
    }
}

/// <summary>Awaits <paramref name="gate"/> before returning one fixed post — for forcing a deterministic ordering between two sites' workers instead of relying on incidental scheduling, which real async I/O (DB calls, HTTP) makes too unreliable to assert a race's outcome against directly.</summary>
internal sealed class GatedSiteClient(string siteName, BooruPost post, Task gate) : IBooruClient
{
    public string SiteName => siteName;
    public int PageSize => 100;
    public IRateLimiter RateLimiter { get; } = new ImmediateRateLimiter();

    public IAsyncEnumerable<BooruTagCount> ListTagsByCountDescendingAsync(CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public async Task<BooruPostPage> ListPostsAsync(string tagQuery, string? cursor, CancellationToken cancellationToken = default)
    {
        if (cursor is not null)
            return new BooruPostPage([], null);

        await gate;
        return new BooruPostPage([post], null);
    }

    public Task<BooruPost?> GetPostAsync(long postId, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();
}

/// <summary>Like <see cref="GatedSiteClient"/> but returns a fixed multi-post page instead of a single post — for testing whether a site's worker correctly stops partway through a page once its own fairness floor is satisfied.</summary>
internal sealed class GatedMultiPostSiteClient(string siteName, IReadOnlyList<BooruPost> posts, Task gate) : IBooruClient
{
    public string SiteName => siteName;
    public int PageSize => 100;
    public IRateLimiter RateLimiter { get; } = new ImmediateRateLimiter();

    public IAsyncEnumerable<BooruTagCount> ListTagsByCountDescendingAsync(CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public async Task<BooruPostPage> ListPostsAsync(string tagQuery, string? cursor, CancellationToken cancellationToken = default)
    {
        if (cursor is not null)
            return new BooruPostPage([], null);

        await gate;
        return new BooruPostPage(posts, null);
    }

    public Task<BooruPost?> GetPostAsync(long postId, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();
}

/// <summary>Answers every request with the same image bytes after an artificial delay — long enough that two concurrent site workers' downloads are reliably still in flight at the same time, widening the race window the download-then-recheck-then-commit pattern (<see cref="DatasetCrawler"/>'s <c>DownloadPostAsync</c>/<c>ProcessDownloadedPostAsync</c>) has to handle correctly.</summary>
internal sealed class DelayedImageHttpMessageHandler(byte[] imageBytes, TimeSpan delay) : HttpMessageHandler
{
    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
        return new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(imageBytes) };
    }
}

/// <summary>Like <see cref="RoutedImageHttpMessageHandler"/> (distinct bytes per URL, so posts don't perceptually dedup against each other) but with a per-request delay that varies (cycling short/long), so a page's posts genuinely finish out of launch order instead of all landing in one simultaneous wave — needed to reliably exercise a "later-launched but earlier-finishing post's progress report gets overwritten by an earlier-launched but later-finishing one" race.</summary>
internal sealed class JitteredDelayRoutedImageHttpMessageHandler(IReadOnlyDictionary<Uri, byte[]> imageBytesByUrl) : HttpMessageHandler
{
    private int _requests;

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var n = Interlocked.Increment(ref _requests);
        await Task.Delay(TimeSpan.FromMilliseconds((n * 7) % 30), cancellationToken).ConfigureAwait(false);
        return new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(imageBytesByUrl[request.RequestUri!]) };
    }
}

/// <summary>Throws on the first-page listing call the configured number of times, then succeeds with a fixed post — for exercising <see cref="DatasetCrawler"/>'s retry-after-failure path deterministically.</summary>
internal sealed class FlakySiteClient(string siteName, int failuresBeforeSuccess, BooruPost post) : IBooruClient
{
    private int _attempts;

    public string SiteName => siteName;
    public int PageSize => 100;
    public IRateLimiter RateLimiter { get; } = new ImmediateRateLimiter();

    public IAsyncEnumerable<BooruTagCount> ListTagsByCountDescendingAsync(CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public async Task<BooruPostPage> ListPostsAsync(string tagQuery, string? cursor, CancellationToken cancellationToken = default)
    {
        await Task.Yield(); // a genuine async yield so a concurrent site's task gets a fair chance to run too
        if (cursor is not null)
            return new BooruPostPage([], null);

        var attempt = Interlocked.Increment(ref _attempts);
        if (attempt <= failuresBeforeSuccess)
            throw new HttpRequestException($"simulated failure (attempt {attempt})");

        return new BooruPostPage([post], null);
    }

    public Task<BooruPost?> GetPostAsync(long postId, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();
}

/// <summary>Answers with different image bytes depending on the request URL — for giving two sites' posts genuinely distinct content (and therefore distinct perceptual hashes) instead of colliding as near-duplicates of each other.</summary>
internal sealed class RoutedImageHttpMessageHandler(IReadOnlyDictionary<Uri, byte[]> imageBytesByUrl) : HttpMessageHandler
{
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
        Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(imageBytesByUrl[request.RequestUri!]) });
}

/// <summary>Like <see cref="RoutedImageHttpMessageHandler"/> but some URLs answer with an explicit HTTP status (no body) instead of real image bytes — for testing that a permanently-dead post's file (404, 410, ...) is skipped rather than retried forever or treated as a site-wide failure.</summary>
internal sealed class RoutedStatusHttpMessageHandler(IReadOnlyDictionary<Uri, byte[]> imageBytesByUrl, IReadOnlyDictionary<Uri, HttpStatusCode> statusByUrl) : HttpMessageHandler
{
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var uri = request.RequestUri!;
        if (statusByUrl.TryGetValue(uri, out var status))
            return Task.FromResult(new HttpResponseMessage(status));
        return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(imageBytesByUrl[uri]) });
    }
}

/// <summary>Fails every listing call, forever — for verifying a persistently broken site doesn't stop the rest of a run from making progress.</summary>
internal sealed class AlwaysFailingSiteClient(string siteName) : IBooruClient
{
    public string SiteName => siteName;
    public int PageSize => 100;
    public IRateLimiter RateLimiter { get; } = new ImmediateRateLimiter();

    public IAsyncEnumerable<BooruTagCount> ListTagsByCountDescendingAsync(CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public async Task<BooruPostPage> ListPostsAsync(string tagQuery, string? cursor, CancellationToken cancellationToken = default)
    {
        await Task.Yield(); // see FlakySiteClient — same reasoning
        throw new HttpRequestException("simulated persistent failure");
    }

    public Task<BooruPost?> GetPostAsync(long postId, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();
}

public class DatasetCrawlerConcurrencyTests
{
    private static byte[] MakeTestPng(byte r = 200, byte g = 100, byte b = 50)
    {
        using var bitmap = new SKBitmap(new SKImageInfo(8, 8, SKColorType.Rgba8888, SKAlphaType.Unpremul));
        using (var canvas = new SKCanvas(bitmap))
            canvas.Clear(new SKColor(r, g, b, 255));
        using var encoded = bitmap.Encode(SKEncodedImageFormat.Png, 100);
        return encoded.ToArray();
    }

    /// <summary>
    /// A flat solid color (see <see cref="MakeTestPng"/>) has zero spatial-frequency
    /// content, so PerceptualHash's DCT reduces every such image to the exact same
    /// hash regardless of which color — fine for tests that want images to dedup
    /// against each other, but useless for a test that needs many genuinely distinct
    /// images. This offsets a couple of solid shapes by <paramref name="seed"/> to give
    /// each variant real low-frequency structure the DCT actually picks up (same trick
    /// PerceptualHashTests.MakeGradientBitmap uses).
    /// </summary>
    private static byte[] MakeDistinctTestPng(int seed)
    {
        using var bitmap = new SKBitmap(new SKImageInfo(64, 64, SKColorType.Rgba8888, SKAlphaType.Unpremul));
        using (var canvas = new SKCanvas(bitmap))
        {
            canvas.Clear(SKColors.White);
            using var shapePaint = new SKPaint { Color = new SKColor(200, 30, 30) };
            canvas.DrawCircle(8 + (seed % 48), 8 + ((seed * 7) % 48), 6, shapePaint);
            canvas.DrawRect(new SKRect(48 - (seed % 40), 48, 56 - (seed % 40), 56), shapePaint);
        }
        using var encoded = bitmap.Encode(SKEncodedImageFormat.Png, 100);
        return encoded.ToArray();
    }

    /// <summary>
    /// Regression test for the exact class of corruption this project spent real effort
    /// diagnosing and repairing on a live dataset: two different (site, post) pairs that
    /// turn out to be the same underlying image, discovered at nearly the same instant
    /// by two concurrent site workers. Before ProcessPostAsync's lock was split into a
    /// fast dedup check, an unlocked download, and a re-checked commit, two sites
    /// racing on the same md5 could both pass the "not yet known" check and both append
    /// — producing two Images rows for one image, one of them holding a stale/wrong
    /// CacheRowIndex association exactly like the corruption already fixed on the real
    /// dataset earlier. This asserts that can't happen: exactly one Images row, with
    /// both sites recorded as sources of it.
    /// </summary>
    [Fact]
    public async Task RunAsync_TwoSitesDiscoveringTheSameImageConcurrently_CommitsExactlyOneImageRow()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            await db.UpsertTagSurveysAsync([("1girl", (int?)10, (int?)10, true)], DateTimeOffset.UtcNow, null);

            var imageBytes = MakeTestPng();
            var fileUrl = new Uri("https://example.test/img.png");

            var danbooruPost = new BooruPost(1, "same-image-both-sites", fileUrl, ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8);
            var gelbooruPost = new BooruPost(2, "same-image-both-sites", fileUrl, ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8);

            var clients = new Dictionary<string, IBooruClient>
            {
                ["danbooru"] = new ConcurrencyTestClient("danbooru", danbooruPost),
                ["gelbooru"] = new ConcurrencyTestClient("gelbooru", gelbooruPost),
            };

            using var downloadClient = new HttpClient(new DelayedImageHttpMessageHandler(imageBytes, TimeSpan.FromMilliseconds(150)));

            // minImages/maxImages=1 and negativeTarget=0 keep this to exactly one page,
            // one post, per site — the smallest run that still exercises two sites
            // racing on the very first image either of them ever sees.
            var result = await DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize: 8,
                minImages: 1, maxImages: 1, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress: null, CancellationToken.None);

            Assert.Empty(result.Shortfalls);

            var images = await db.GetAllImagesAsync();
            Assert.Single(images); // the race must never produce two rows for the same image

            var sources = await db.GetImageSourceSnapshotsAsync("same-image-both-sites");
            Assert.Equal(2, sources.Count); // both sites still recorded as provenance
            Assert.Contains(sources, s => s.Site == "danbooru");
            Assert.Contains(sources, s => s.Site == "gelbooru");
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task RunAsync_TranslatesRawPostTagsToTheirSurveyedCategoryIdentity()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            // The survey records a tag's identity — its raw booru name prefixed with
            // its category (see TagCategoryNaming) — not the raw name itself.
            await db.UpsertTagSurveysAsync([("character:frieren", (int?)10, (int?)null, true)], DateTimeOffset.UtcNow, null);

            var imageBytes = MakeTestPng();
            // A site's own API only ever returns a post's tags by their raw name.
            var post = new BooruPost(1, "frieren-image", new Uri("https://example.test/img.png"), ["frieren"], "g", DateTimeOffset.UtcNow, 8, 8);
            var clients = new Dictionary<string, IBooruClient> { ["danbooru"] = new ConcurrencyTestClient("danbooru", post) };

            using var downloadClient = new HttpClient(new DelayedImageHttpMessageHandler(imageBytes, TimeSpan.Zero));

            var result = await DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize: 8,
                minImages: 1, maxImages: 1, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress: null, CancellationToken.None);

            Assert.Empty(result.Shortfalls);

            var combinedCounts = await db.GetAllCombinedPositiveCountsAsync();
            Assert.Equal(1, combinedCounts.GetValueOrDefault("character:frieren"));
            Assert.False(combinedCounts.ContainsKey("frieren")); // never recorded under its raw name
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task RunAsync_ResolvesAliasedRawTagName_ToItsMergedIdentity_WhenTagAliasesProvided()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            // Simulates the post-merge survey state: only "headpat" survived as its own
            // eligible tag row (TagSurveyor folds "head_pat" into it before this point) —
            // "head_pat" is NOT a separate eligible tag anymore.
            await db.UpsertTagSurveysAsync([("headpat", (int?)10, (int?)10, true)], DateTimeOffset.UtcNow, null);

            var imageBytes = MakeTestPng();
            // Gelbooru's own posts keep using its own raw spelling forever — it has no
            // idea Danbooru aliased "head_pat" to "headpat".
            var post = new BooruPost(1, "head-pat-image", new Uri("https://example.test/img.png"), ["head_pat"], "g", DateTimeOffset.UtcNow, 8, 8);
            var clients = new Dictionary<string, IBooruClient> { ["gelbooru"] = new ConcurrencyTestClient("gelbooru", post) };
            var tagAliases = new Dictionary<string, string> { ["head_pat"] = "headpat" };

            using var downloadClient = new HttpClient(new DelayedImageHttpMessageHandler(imageBytes, TimeSpan.Zero));

            var result = await DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize: 8,
                minImages: 1, maxImages: 1, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress: null, CancellationToken.None, tagAliases: tagAliases);

            Assert.Empty(result.Shortfalls);

            var combinedCounts = await db.GetAllCombinedPositiveCountsAsync();
            Assert.Equal(1, combinedCounts.GetValueOrDefault("headpat"));
            Assert.False(combinedCounts.ContainsKey("head_pat")); // never recorded under the aliased-away raw name
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task RunAsync_WithoutTagAliases_SilentlyDropsAnAliasedAwayRawTagName()
    {
        // Regression guard for the bug the previous test's fix corrected: without a
        // tagAliases map, an already-merged tag's antecedent raw name isn't in
        // eligibleTagIdentities at all anymore (TagSurveyor folded it away at survey
        // time), so a post still using that raw spelling loses the tag entirely instead
        // of crediting either identity — worse than before the merge, which at least
        // recorded it under the wrong name.
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            await db.UpsertTagSurveysAsync([("headpat", (int?)10, (int?)10, true)], DateTimeOffset.UtcNow, null);

            var imageBytes = MakeTestPng();
            var post = new BooruPost(1, "head-pat-image-no-alias", new Uri("https://example.test/img.png"), ["head_pat"], "g", DateTimeOffset.UtcNow, 8, 8);
            var clients = new Dictionary<string, IBooruClient> { ["gelbooru"] = new ConcurrencyTestClient("gelbooru", post) };

            using var downloadClient = new HttpClient(new DelayedImageHttpMessageHandler(imageBytes, TimeSpan.Zero));

            await DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize: 8,
                minImages: 1, maxImages: 1, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress: null, CancellationToken.None); // no tagAliases passed

            // GetAllCombinedPositiveCountsAsync returns every surveyed tag row, including
            // untouched ones at their default 0 — "headpat" being a KEY is expected (it's
            // the one surveyed tag); the bug this guards is its VALUE never moving.
            var combinedCounts = await db.GetAllCombinedPositiveCountsAsync();
            Assert.Equal(0, combinedCounts.GetValueOrDefault("headpat"));
            Assert.Equal(0, combinedCounts.GetValueOrDefault("head_pat"));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task RunAsync_RetriesAfterAFailure_AndLogsEachAttempt()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            await db.UpsertTagSurveysAsync([("1girl", (int?)10, (int?)null, true)], DateTimeOffset.UtcNow, null);

            var imageBytes = MakeTestPng();
            var post = new BooruPost(1, "flaky-retry-image", new Uri("https://example.test/img.png"), ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8);
            var clients = new Dictionary<string, IBooruClient> { ["danbooru"] = new FlakySiteClient("danbooru", failuresBeforeSuccess: 2, post) };

            using var downloadClient = new HttpClient(new DelayedImageHttpMessageHandler(imageBytes, TimeSpan.Zero));

            var result = await DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize: 8,
                minImages: 1, maxImages: 1, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress: null, CancellationToken.None,
                retryDelay: (_, _) => Task.CompletedTask); // no real 20-minute wait in a test

            Assert.Empty(result.Shortfalls);

            var images = await db.GetAllImagesAsync();
            Assert.Single(images); // eventually succeeded despite the first two failures

            var logPath = CrawlErrorLog.ForDirectory(directory).LogPath;
            Assert.True(File.Exists(logPath));
            var logLines = await File.ReadAllLinesAsync(logPath);
            Assert.Equal(2, logLines.Length); // one durable line per failed attempt
            Assert.All(logLines, line => Assert.Contains("danbooru", line));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>
    /// Regression test for the deadlock risk a "never give up" retry policy introduces
    /// if paired with a cross-site barrier: before RunSiteWorkerAsync folded each site's
    /// positive-then-negative work into one independent worker, a site stuck retrying
    /// forever inside a Task.WhenAll barrier would have blocked every other site from
    /// ever reaching the negative phase, hanging the whole run.
    ///
    /// danbooru here never succeeds even once, so — correctly, given the per-site
    /// fairness floor (see RunAsync's own doc comment) — its worker never stops
    /// retrying either; a broken site that never once got its fair share is exactly the
    /// case that floor exists to keep searching for, not the case to silently give up
    /// on. So this test doesn't wait for RunAsync itself to return (it won't, by
    /// design); it polls crawl.sqlite directly for gelbooru's contribution to land,
    /// proving gelbooru's own worker isn't stuck behind danbooru's, then cancels.
    /// </summary>
    [Fact]
    public async Task RunAsync_OneSitePersistentlyFailing_DoesNotBlockTheOtherSite()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            await db.UpsertTagSurveysAsync([("1girl", (int?)10, (int?)10, true)], DateTimeOffset.UtcNow, null);

            var imageBytes = MakeTestPng();
            var gelbooruPost = new BooruPost(1, "gelbooru-only-image", new Uri("https://example.test/img.png"), ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8);

            var clients = new Dictionary<string, IBooruClient>
            {
                ["danbooru"] = new AlwaysFailingSiteClient("danbooru"),
                ["gelbooru"] = new ConcurrencyTestClient("gelbooru", gelbooruPost),
            };

            using var downloadClient = new HttpClient(new DelayedImageHttpMessageHandler(imageBytes, TimeSpan.Zero));
            using var cts = new CancellationTokenSource();

            var runTask = DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize: 8,
                minImages: 1, maxImages: 1, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress: null, cts.Token,
                retryDelay: (_, _) => Task.CompletedTask);

            var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(10);
            IReadOnlyList<(string Md5, int CacheRowIndex, ulong PHash)> images = [];
            while (DateTime.UtcNow < deadline && images.Count == 0)
            {
                images = await db.GetAllImagesAsync();
                if (images.Count == 0)
                    await Task.Delay(20);
            }

            Assert.Single(images);
            Assert.Equal("gelbooru-only-image", images[0].Md5); // landed despite danbooru never working

            var logPath = CrawlErrorLog.ForDirectory(directory).LogPath;
            Assert.True(File.Exists(logPath));
            Assert.Contains(await File.ReadAllLinesAsync(logPath), line => line.Contains("danbooru"));

            cts.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => runTask);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>
    /// Regression test for the exact bug reported against a live crawl: a corpus where
    /// gelbooru had contributed to zero images as the sole discoverer, despite clearly
    /// making requests the whole time — because danbooru alone could already satisfy a
    /// tag's shared --max-images quota before gelbooru's own (slower) request even
    /// landed, so gelbooru's shouldContinue() check came up false before it ever got a
    /// real look. Here maxImages=1 with two sites means danbooru's single post already
    /// meets the combined target on its own — without RunAsync's per-site fairness
    /// floor, gelbooru's exclusive image would never be found. With it (each site
    /// floors at ceil(maxImages/siteCount) of its own contributions, independent of
    /// what the other site already found), gelbooru still searches and its own
    /// exclusive image lands too.
    /// </summary>
    [Fact]
    public async Task RunAsync_PerSiteFairnessFloor_LetsASlowerSiteStillContributeItsOwnExclusiveImage()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            await db.UpsertTagSurveysAsync([("1girl", (int?)10, (int?)10, true)], DateTimeOffset.UtcNow, null);

            var danbooruUrl = new Uri("https://example.test/danbooru.png");
            var gelbooruUrl = new Uri("https://example.test/gelbooru.png");
            var imageBytesByUrl = new Dictionary<Uri, byte[]>
            {
                [danbooruUrl] = MakeTestPng(200, 100, 50),
                [gelbooruUrl] = MakeTestPng(10, 220, 90), // visibly different content -> a different perceptual hash, not a near-duplicate of the other
            };

            var danbooruPost = new BooruPost(1, "danbooru-only-image", danbooruUrl, ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8);
            var gelbooruPost = new BooruPost(2, "gelbooru-only-image", gelbooruUrl, ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8);

            // gelbooru's listing call deliberately doesn't return until danbooru's image
            // is confirmed durably committed below — real async I/O (SQLite, HTTP) makes
            // incidental scheduling order too unreliable to assert this test's outcome
            // against otherwise; an earlier version of this test passed identically with
            // and without the fairness fix in place, which is exactly the trap this gate
            // avoids.
            var gelbooruGate = new TaskCompletionSource();
            var clients = new Dictionary<string, IBooruClient>
            {
                ["danbooru"] = new ConcurrencyTestClient("danbooru", danbooruPost),
                ["gelbooru"] = new GatedSiteClient("gelbooru", gelbooruPost, gelbooruGate.Task),
            };

            using var downloadClient = new HttpClient(new RoutedImageHttpMessageHandler(imageBytesByUrl));

            // maxImages=1: danbooru's own single post already meets the combined
            // target by itself — the exact scenario that starved gelbooru for real.
            var runTask = DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize: 8,
                minImages: 1, maxImages: 1, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress: null, CancellationToken.None);

            var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(10);
            while (DateTime.UtcNow < deadline && !(await db.GetAllImagesAsync()).Any(i => i.Md5 == "danbooru-only-image"))
                await Task.Delay(10);
            gelbooruGate.SetResult();

            var result = await runTask;

            var images = await db.GetAllImagesAsync();
            Assert.Equal(2, images.Count); // both sites' exclusive finds made it in, not just the faster one's
            Assert.Contains(images, i => i.Md5 == "danbooru-only-image");
            Assert.Contains(images, i => i.Md5 == "gelbooru-only-image");
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>
    /// Regression test for a live dataset that had 557 images sitting in images.bin/
    /// tag_rows.jsonl with no crawl.sqlite record at all — traced to a checkpoint's
    /// Flush() succeeding (making the cache file durable) right before an exit request
    /// (Ctrl+C) landed and CommitPendingImagesAsync threw OperationCanceledException
    /// before it could run, leaving the cache ahead of the database with no way to
    /// recover the gap's provenance. Here a page of two posts has one download
    /// cancelled mid-flight (simulating an exit request landing mid-page).
    ///
    /// Posts within a page now run concurrently (see DatasetCrawler's own doc comment),
    /// so which of post1/post2 — if either — actually finishes before the cancellation
    /// lands isn't deterministic the way it was when posts were processed strictly one
    /// at a time; asserting a specific post survived would be asserting scheduling
    /// behavior, not a real guarantee. What must still hold, regardless of how many
    /// posts made it through, is the actual invariant this regression guards: whatever
    /// the cache file durably has, crawl.sqlite durably knows about too — checked here
    /// with the same <see cref="CacheConsistency.Validate"/> a resumed run relies on.
    /// </summary>
    [Fact]
    public async Task RunAsync_CancelledMidPage_CacheAndDatabaseStayConsistent()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            await db.UpsertTagSurveysAsync([("1girl", (int?)10, (int?)null, true)], DateTimeOffset.UtcNow, null);

            var post1 = new BooruPost(1, "maybe-appended-before-cancel-1", new Uri("https://example.test/1.png"), ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8);
            var post2 = new BooruPost(2, "maybe-appended-before-cancel-2", new Uri("https://example.test/2.png"), ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8);

            var clients = new Dictionary<string, IBooruClient> { ["danbooru"] = new MultiPostSiteClient("danbooru", [post1, post2]) };

            using var cts = new CancellationTokenSource();
            // Cancels on the 2nd HTTP request, whichever post that turns out to be.
            using var downloadClient = new HttpClient(new CancelOnRequestHttpMessageHandler(MakeTestPng(), cancelOnRequestNumber: 2, cts));

            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize: 8,
                minImages: 1, maxImages: 3, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress: null, cts.Token));

            var images = await db.GetAllImagesAsync();
            using var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(directory, inputSize: 8);

            CacheConsistency.Validate(images, writer.ImageCount, directory); // throws if crawl.sqlite references a cache row that doesn't exist
            Assert.Equal(writer.ImageCount, images.Count); // ...and no cache-only row crawl.sqlite doesn't know about either
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>
    /// Regression test for the fairness-floor gap this project's own progress bars
    /// surfaced live: a duplicate post skipped only because the canonical image
    /// ALREADY carries the tag this site is searching for is still real, correct
    /// work — the site found a genuinely matching post, it just happened to already
    /// be known. Before MergeDuplicateTags credited SitePositiveCounts for that case
    /// too, only brand-new tag-image associations counted, so a site whose search
    /// results page turned out to be entirely redundant with what's already known got
    /// zero fairness-floor credit for any of it despite searching correctly — the same
    /// "stuck at 0/floor, still grinding" symptom reported against a live crawl, via a
    /// different path than <see cref="RunAsync_PerSiteFairnessFloor_LetsASlowerSiteStillContributeItsOwnExclusiveImage"/>
    /// covers (that one is about a site never even getting a look; this one is about a
    /// site looking and finding only redundant matches).
    ///
    /// Both sites share a floor of 1 (maxImages=1, two sites). danbooru's post is
    /// exclusive and lands first (gated so gelbooru only starts once it's durably
    /// committed). gelbooru's own page then returns two posts: the first is an exact-
    /// md5 duplicate of danbooru's already-known image, still tagged "1girl" — the
    /// redundant-but-relevant case; the second is a genuinely new, gelbooru-exclusive
    /// image. With the fix, gelbooru's floor is satisfied by the first (redundant)
    /// post alone, so it correctly stops before ever looking at the second — exactly
    /// one image total. Without the fix, the redundant post credits nothing, gelbooru's
    /// floor stays unmet, and it goes on to find and append its second, exclusive
    /// image too — two images total. That difference is what this test asserts.
    /// </summary>
    [Fact]
    public async Task RunAsync_RedundantDuplicateMatchingSearchedTag_StillCountsTowardSiteFloor()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            await db.UpsertTagSurveysAsync([("1girl", (int?)10, (int?)10, true)], DateTimeOffset.UtcNow, null);

            var sharedUrl = new Uri("https://example.test/shared.png");
            var exclusiveUrl = new Uri("https://example.test/exclusive.png");
            var imageBytesByUrl = new Dictionary<Uri, byte[]>
            {
                [sharedUrl] = MakeTestPng(200, 100, 50),
                [exclusiveUrl] = MakeTestPng(10, 220, 90), // visibly different content -> distinct perceptual hash, not a near-duplicate of the shared one
            };

            var danbooruPost = new BooruPost(1, "shared-image", sharedUrl, ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8);
            // Same md5 as danbooru's post — gelbooru's own search re-listing the same
            // underlying image, already carrying "1girl", not a new one of its own.
            var gelbooruRedundantPost = new BooruPost(2, "shared-image", sharedUrl, ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8);
            var gelbooruExclusivePost = new BooruPost(3, "gelbooru-exclusive-image", exclusiveUrl, ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8);

            var gelbooruGate = new TaskCompletionSource();
            var clients = new Dictionary<string, IBooruClient>
            {
                ["danbooru"] = new ConcurrencyTestClient("danbooru", danbooruPost),
                ["gelbooru"] = new GatedMultiPostSiteClient("gelbooru", [gelbooruRedundantPost, gelbooruExclusivePost], gelbooruGate.Task),
            };

            using var downloadClient = new HttpClient(new RoutedImageHttpMessageHandler(imageBytesByUrl));

            // maxImages=1: once danbooru's post lands, the combined target is already
            // met — only the per-site floor (unmet at 0/1 for gelbooru) can still be
            // keeping gelbooru's worker going into its second post.
            var runTask = DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize: 8,
                minImages: 1, maxImages: 1, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress: null, CancellationToken.None);

            var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(10);
            while (DateTime.UtcNow < deadline && !(await db.GetAllImagesAsync()).Any(i => i.Md5 == "shared-image"))
                await Task.Delay(10);
            gelbooruGate.SetResult();

            var result = await runTask;

            Assert.Empty(result.Shortfalls);

            var images = await db.GetAllImagesAsync();
            Assert.Single(images); // gelbooru's floor was met by the redundant match alone; it never reached its exclusive post
            Assert.Equal("shared-image", images[0].Md5);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>
    /// Regression test for a permanently-dead post (image deleted/moved on the site,
    /// download returns 404/410) previously being treated as a SITE-wide failure:
    /// EnsureSuccessStatusCode's exception propagated up to RunSiteTagPhaseAsync's
    /// per-post catch, which aborts the rest of the page, waits (in production)
    /// SiteRetryDelay, and retries the exact same page — including the exact same
    /// permanently-dead post — forever, without ever making progress past it (verified
    /// by hand: sabotaging the fix made this exact test hang indefinitely rather than
    /// just fail, since nothing ever advances the cursor past the dead post — hence
    /// the bounded CancellationTokenSource below instead of CancellationToken.None,
    /// so a regression here fails fast instead of hanging the whole suite). 404/410
    /// are never going to succeed no matter how many times they're retried, so they
    /// must just skip that one post instead. Here a page has one dead (404) post
    /// followed by one live one — the live post must still get processed (proving the
    /// page didn't abort), and nothing should land in crawl-errors.log (proving it
    /// wasn't treated as a site failure at all).
    /// </summary>
    [Fact]
    public async Task RunAsync_PostWithA404Download_SkipsItWithoutTreatingItAsASiteFailure()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            await db.UpsertTagSurveysAsync([("1girl", (int?)10, (int?)null, true)], DateTimeOffset.UtcNow, null);

            var deadUrl = new Uri("https://example.test/dead.png");
            var liveUrl = new Uri("https://example.test/live.png");
            var deadPost = new BooruPost(1, "dead-post-404", deadUrl, ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8);
            var livePost = new BooruPost(2, "live-post", liveUrl, ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8);

            var clients = new Dictionary<string, IBooruClient> { ["danbooru"] = new MultiPostSiteClient("danbooru", [deadPost, livePost]) };

            using var downloadClient = new HttpClient(new RoutedStatusHttpMessageHandler(
                new Dictionary<Uri, byte[]> { [liveUrl] = MakeTestPng() },
                new Dictionary<Uri, HttpStatusCode> { [deadUrl] = HttpStatusCode.NotFound }));

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
            var result = await DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize: 8,
                minImages: 1, maxImages: 1, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress: null, cts.Token,
                retryDelay: (_, _) => Task.CompletedTask); // no real 20-minute wait, and none should be needed anyway

            Assert.Empty(result.Shortfalls);

            var images = await db.GetAllImagesAsync();
            Assert.Single(images); // only the live post made it in — the dead one was skipped, not retried
            Assert.Equal("live-post", images[0].Md5);

            var logPath = CrawlErrorLog.ForDirectory(directory).LogPath;
            Assert.False(File.Exists(logPath)); // a 404 must NOT be treated as a site-wide failure
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>
    /// Companion to <see cref="RunAsync_PostWithA404Download_SkipsItWithoutTreatingItAsASiteFailure"/>:
    /// 401 must NOT be silently skipped like 404/410 — it usually means something is
    /// wrong with OUR request (bad/expired credentials, an IP block, ...), which is
    /// systemic and worth surfacing/retrying like any other site failure rather than
    /// swallowed post-by-post while the actual cause goes unnoticed. This doesn't wait
    /// for RunAsync to return — a persistent 401 correctly retries forever, by design,
    /// same reasoning as <see cref="RunAsync_OneSitePersistentlyFailing_DoesNotBlockTheOtherSide"/>
    /// — it polls crawl-errors.log for the failure to land, then cancels.
    /// </summary>
    [Fact]
    public async Task RunAsync_PostWithA401Download_StillTreatsItAsASiteFailure()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            await db.UpsertTagSurveysAsync([("1girl", (int?)10, (int?)null, true)], DateTimeOffset.UtcNow, null);

            var unauthorizedUrl = new Uri("https://example.test/unauthorized.png");
            var post = new BooruPost(1, "unauthorized-post", unauthorizedUrl, ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8);
            var clients = new Dictionary<string, IBooruClient> { ["danbooru"] = new ConcurrencyTestClient("danbooru", post) };

            using var downloadClient = new HttpClient(new RoutedStatusHttpMessageHandler(
                new Dictionary<Uri, byte[]>(),
                new Dictionary<Uri, HttpStatusCode> { [unauthorizedUrl] = HttpStatusCode.Unauthorized }));

            using var cts = new CancellationTokenSource();
            var runTask = DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize: 8,
                minImages: 1, maxImages: 1, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress: null, cts.Token,
                // A real (if tiny) delay, not Task.CompletedTask: ConcurrencyTestClient's
                // ListPostsAsync returns an already-completed Task, so with an instantly-
                // completing retryDelay too, a persistent failure here never actually
                // yields back to the scheduler — a tight synchronous loop that starves
                // this test's own polling below instead of a clean, fast failure.
                retryDelay: (_, ct) => Task.Delay(1, ct));

            var logPath = CrawlErrorLog.ForDirectory(directory).LogPath;
            var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(10);
            while (DateTime.UtcNow < deadline && !File.Exists(logPath))
                await Task.Delay(20);

            Assert.True(File.Exists(logPath)); // a 401 MUST be treated as a site-wide failure, not silently skipped
            Assert.Contains(await File.ReadAllLinesAsync(logPath), line => line.Contains("401"));

            var images = await db.GetAllImagesAsync();
            Assert.Empty(images); // never silently skipped/appended past the failure

            cts.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => runTask);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>
    /// Regression test for a live-run report: with posts inside a page processed
    /// concurrently, the download/processing bars and the "current tag" count would
    /// visibly get stuck below the true value (e.g. "11/195" forever) even though
    /// checkpointing had already correctly moved on — not an actual stall, but two (or
    /// more) posts' progress reports racing to write the same on-screen counter, where a
    /// post that read an earlier, lower value could still win the write after a post that
    /// read a later, higher one. <c>progressLock</c> in <c>RunSiteTagPhaseAsync</c> (and
    /// moving the global images-appended report inside <c>ProcessDownloadedPostAsync</c>'s
    /// own <c>stateLock</c> section) fixes this by serializing the read-then-display step,
    /// not just the underlying counters (which were already correct on their own).
    ///
    /// Asserted here by recording every progress call through a fake
    /// <see cref="SiteProgressReporter"/> and checking the sequence never regresses
    /// within a page, and that the final report for each row matches the page's true
    /// final count instead of some earlier, lower snapshot a race let win.
    /// </summary>
    [Fact]
    public async Task RunAsync_ProgressReporting_NeverRegressesBelowAPriorValue_UnderPageConcurrency()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            await db.UpsertTagSurveysAsync([("1girl", (int?)50, (int?)null, true)], DateTimeOffset.UtcNow, null);

            const int postCount = 40;
            var posts = Enumerable.Range(0, postCount)
                .Select(i => new BooruPost(i, $"img-{i}", new Uri($"https://example.test/{i}.png"), ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8))
                .ToList();
            var clients = new Dictionary<string, IBooruClient> { ["danbooru"] = new MultiPostSiteClient("danbooru", posts) };

            // Distinct (non-perceptually-duplicate) bytes per post — a flat solid color
            // would dedup every post down to one image (PerceptualHash reduces any flat
            // color to the same hash — see MakeDistinctTestPng), which is wrong for this
            // test: it needs all 40 to genuinely commit, not merge into each other.
            var imageBytesByUrl = posts.ToDictionary(p => p.FileUrl, p => MakeDistinctTestPng((int)p.PostId));

            // Varying per-request delay so posts genuinely complete out of launch order —
            // a fixed/zero delay would let them all land in one simultaneous wave, which
            // exercises the race but doesn't distinguish "a later post's report legitimately
            // overtakes an earlier one" from "an earlier post's stale report wins by luck".
            using var downloadClient = new HttpClient(new JitteredDelayRoutedImageHttpMessageHandler(imageBytesByUrl));

            var reportLock = new object();
            var downloadReports = new List<(int Completed, int Total)>();
            var processingReports = new List<(int Completed, int Total)>();
            var tagReports = new List<long>();

            var siteReporter = new SiteProgressReporter(
                ReportPhase: _ => { },
                ReportDownloadProgress: (completed, total) => { lock (reportLock) downloadReports.Add((completed, total)); },
                ReportProcessingProgress: (completed, total) => { lock (reportLock) processingReports.Add((completed, total)); },
                ReportTagProgress: (_, completed, _, _) => { lock (reportLock) tagReports.Add(completed); },
                ReportTagsCompleted: (_, _) => { });

            var progress = new CrawlProgressReporter(
                new Dictionary<string, SiteProgressReporter> { ["danbooru"] = siteReporter },
                (_, _) => { },
                () => { });

            var result = await DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize: 8,
                minImages: 1, maxImages: postCount, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress, CancellationToken.None);

            Assert.Empty(result.Shortfalls);

            var images = await db.GetAllImagesAsync();
            Assert.Equal(postCount, images.Count); // every post was a genuinely distinct image, so all must have committed

            // A drop to 0 mid-sequence is the legitimate reset for a new page, not a
            // regression — anything else going backwards means a stale read won a race
            // against a fresher one.
            AssertNeverRegressesExceptOnReset(downloadReports.Select(r => (long)r.Completed).ToList());
            AssertNeverRegressesExceptOnReset(processingReports.Select(r => (long)r.Completed).ToList());
            AssertNeverRegressesExceptOnReset(tagReports);

            Assert.Equal(postCount, downloadReports[^1].Completed);
            Assert.Equal(postCount, processingReports[^1].Completed);

            // Not tagReports[^1]: with negativeTarget=0 the negative phase's own initial
            // "current tag" report (0 negatives needed against a target of 0) legitimately
            // follows the positive phase's — a genuine reset this test's regression check
            // already tolerates, not the value this assertion cares about. What matters is
            // that the positive phase's own count actually reached postCount at some point
            // rather than a stale, lower snapshot winning the last write.
            Assert.Contains((long)postCount, tagReports);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>
    /// A live run reported "current tag" staying at 0 no matter how many posts were
    /// processed, unchanged across repeated observations of the same tag — not the
    /// symptom of a display race (which would show some fluctuation). This exercises the
    /// actual production shape: a large pre-existing corpus where searching a tag for the
    /// FIRST time immediately turns up images already known from other tags' earlier
    /// searches (or an import) — not a stale, already-exhausted tag being re-searched.
    ///
    /// The image is seeded directly into crawl.sqlite/the cache (bypassing RunAsync)
    /// specifically so this tag's own (tag, site, phase) progress record stays untouched
    /// — a first attempt at this test ran RunAsync twice instead, which instead exercised
    /// (and passed past, uninterestingly) the unrelated "this tag's pagination is already
    /// exhausted" short-circuit at the top of RunSiteTagPhaseAsync's while loop, never
    /// reaching a single post.
    /// </summary>
    [Fact]
    public async Task RunAsync_RediscoveringAnAlreadyKnownImage_StillCreditsTheSearchedTag()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            const int inputSize = 8;

            var vocabulary = TagVocabulary.CreateEmpty();
            vocabulary.AddTag("head_pat"); // row 0
            vocabulary.Save(Path.Combine(directory, "tag_vocabulary.json"));

            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize))
                writer.Append(new EncodedImage(Enumerable.Range(0, inputSize * inputSize * 3).Select(i => (byte)i).ToArray(), new LetterboxBox(0, 0, inputSize, inputSize)), [0]);

            using var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            await db.UpsertTagSurveysAsync([("head_pat", (int?)10, (int?)null, true)], DateTimeOffset.UtcNow, null);

            var seedImage = new PendingNewImage(
                "already-known-image", 0, 100, 100, DateTimeOffset.UtcNow,
                "danbooru", 999, "https://example.test/999.png", "g", DateTimeOffset.UtcNow,
                ["head_pat"], PHash: 0);
            await db.CommitPendingImagesAsync([seedImage], [], [], CancellationToken.None);

            // A "head_pat" search — never run before against this crawl.sqlite — finds this
            // same image again (a different post, same underlying md5). It must dedup-skip
            // rather than append a duplicate row, but should still credit this run's own
            // "current tag" progress for having found it.
            var post = new BooruPost(1, "already-known-image", new Uri("https://example.test/1.png"), ["head_pat"], "g", DateTimeOffset.UtcNow, 8, 8);
            var clients = new Dictionary<string, IBooruClient> { ["danbooru"] = new MultiPostSiteClient("danbooru", [post]) };
            using var downloadClient = new HttpClient(new RoutedImageHttpMessageHandler(new Dictionary<Uri, byte[]> { [post.FileUrl] = MakeDistinctTestPng(seed: 1) }));

            var tagReports = new List<long>();
            var siteReporter = new SiteProgressReporter(
                ReportPhase: _ => { },
                ReportDownloadProgress: (_, _) => { },
                ReportProcessingProgress: (_, _) => { },
                ReportTagProgress: (_, completed, _, _) => tagReports.Add(completed),
                ReportTagsCompleted: (_, _) => { });
            var progress = new CrawlProgressReporter(
                new Dictionary<string, SiteProgressReporter> { ["danbooru"] = siteReporter },
                (_, _) => { },
                () => { });

            var result = await DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize,
                minImages: 1, maxImages: 1, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress, CancellationToken.None);

            Assert.Single(await db.GetAllImagesAsync()); // rediscovery must not append a duplicate row
            Assert.Contains(1L, tagReports); // ...but must still show up in "current tag" progress
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>
    /// Reproduction of the exact live-run report: a resumed process joins a tag
    /// mid-pagination (a prior process instance already durably committed a page's worth
    /// of this tag's images before stopping) and its "current tag" progress must reflect
    /// that real prior work from the moment it starts — not just eventually climb as new
    /// posts get credited (which per-post crediting already did correctly; the actual gap
    /// was SitePositiveCounts having no memory across a restart at all). Checks both: the
    /// very first reported value already shows the resumed site's real prior credit
    /// (seeded in <c>RunSiteTagPhaseAsync</c> directly from <see cref="TagProgressState.SitePositiveCount"/>
    /// — the durable per-page checkpoint a real prior process would have saved — before
    /// that first report), and it climbs further as this process's own page-2 posts —
    /// all rediscoveries of images already known from elsewhere, the realistic shape
    /// once a corpus is large — get processed.
    /// </summary>
    [Fact]
    public async Task RunAsync_ResumedMidPagination_CreditsManyRediscoveredDuplicatesOnTheContinuedPage()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            const int inputSize = 8;
            const int alreadyKnownCount = 30;

            var vocabulary = TagVocabulary.CreateEmpty();
            vocabulary.AddTag("head_pat"); // row 0
            vocabulary.Save(Path.Combine(directory, "tag_vocabulary.json"));

            // Simulates images already known from elsewhere (other tags' earlier
            // searches, or head_pat's own earlier pages from before a restart) — seeded
            // directly rather than by actually running those earlier pages, since only
            // the RESUMED state matters for this test.
            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize))
            {
                for (var i = 0; i < alreadyKnownCount; i++)
                    writer.Append(new EncodedImage(Enumerable.Range(0, inputSize * inputSize * 3).Select(b => (byte)(b + i)).ToArray(), new LetterboxBox(0, 0, inputSize, inputSize)), [0]);
            }

            using var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            await db.UpsertTagSurveysAsync([("head_pat", (int?)1000, (int?)null, true)], DateTimeOffset.UtcNow, null);

            var seedImages = Enumerable.Range(0, alreadyKnownCount)
                .Select(i => new PendingNewImage($"already-known-{i}", i, 100, 100, DateTimeOffset.UtcNow, "danbooru", i, $"https://example.test/known-{i}.png", "g", DateTimeOffset.UtcNow, ["head_pat"], PHash: 0))
                .ToList();
            await db.CommitPendingImagesAsync(seedImages, [], [], CancellationToken.None);

            // The tag/site/phase progress record a real prior process would have saved:
            // partway through pagination, not Done, with SitePositiveCount checkpointed
            // to this site's real prior credit (see Flush's own per-page save) — the
            // exact durable state a resumed process reads on startup and seeds straight
            // from, no ImageSources re-derivation involved.
            await db.SaveTagProgressAsync("head_pat", "danbooru", "positive",
                new TagProgressState("page-2-cursor", alreadyKnownCount, Done: false, SitePositiveCount: alreadyKnownCount),
                CancellationToken.None);

            // Page 2: the same already-known images turning up again — e.g. the search
            // re-lists something already found via a different tag's earlier crawl.
            var page2Posts = Enumerable.Range(0, alreadyKnownCount)
                .Select(i => new BooruPost(1000 + i, $"already-known-{i}", new Uri($"https://example.test/page2-{i}.png"), ["head_pat"], "g", DateTimeOffset.UtcNow, 8, 8))
                .ToList();
            var pagesByCursor = new Dictionary<string?, BooruPostPage> { ["page-2-cursor"] = new BooruPostPage(page2Posts, null) };
            var clients = new Dictionary<string, IBooruClient> { ["danbooru"] = new CursorPagedSiteClient("danbooru", pagesByCursor) };

            var imageBytesByUrl = page2Posts.ToDictionary(p => p.FileUrl, p => MakeDistinctTestPng((int)p.PostId));
            using var downloadClient = new HttpClient(new RoutedImageHttpMessageHandler(imageBytesByUrl));

            var tagReports = new List<long>();
            var siteReporter = new SiteProgressReporter(
                ReportPhase: _ => { },
                ReportDownloadProgress: (_, _) => { },
                ReportProcessingProgress: (_, _) => { },
                ReportTagProgress: (_, completed, _, _) => tagReports.Add(completed),
                ReportTagsCompleted: (_, _) => { });
            var progress = new CrawlProgressReporter(
                new Dictionary<string, SiteProgressReporter> { ["danbooru"] = siteReporter },
                (_, _) => { },
                () => { });

            // maxImages set higher than alreadyKnownCount specifically so the seeded
            // credit alone doesn't already satisfy the per-site floor — otherwise the
            // loop would (correctly!) skip page 2 entirely, which would prove seeding
            // works but never exercise "credit climbs further from here" too.
            var result = await DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize,
                minImages: 1, maxImages: alreadyKnownCount * 2, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress, CancellationToken.None);

            Assert.Equal(alreadyKnownCount, (await db.GetAllImagesAsync()).Count); // no duplicate rows from rediscovery

            // The very first report — before a single page-2 post is processed — must
            // already reflect this site's real prior credit, seeded straight from the
            // persisted TagProgressState.SitePositiveCount, not the pre-fix 0.
            Assert.Equal(alreadyKnownCount, tagReports[0]);

            // ...and it must climb further as page 2's own rediscoveries get credited on
            // top of that seeded baseline.
            Assert.Contains((long)(alreadyKnownCount * 2), tagReports);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task RunAsync_ResumedWithQuotaAlreadySatisfied_SkipsRefetchingEntirely()
    {
        const int inputSize = 8;
        const int maxImages = 3;

        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            await db.UpsertTagSurveysAsync([("1girl", (int?)1000, (int?)null, true)], DateTimeOffset.UtcNow, null);

            // More posts than maxImages needs, on a single page whose NextCursor is
            // deliberately non-null: this site's own pagination is NOT exhausted, so the
            // only way the crawl stops on this tag is shouldContinue() going false
            // partway through the page once the per-site floor (== maxImages, single
            // site) is met -- exactly the "quota satisfied, not Done" case
            // QuotaSatisfiedAtMaxImages exists for, as opposed to the already-covered
            // "genuinely exhausted" (Done == true) case.
            var posts = Enumerable.Range(0, maxImages + 2)
                .Select(i => new BooruPost(i, $"img-{i}", new Uri($"https://example.test/{i}.png"), ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8))
                .ToList();
            var firstPage = new BooruPostPage(posts, "more-pages-exist");
            var client = new CountingCursorPagedSiteClient("danbooru", firstPage);
            var clients = new Dictionary<string, IBooruClient> { ["danbooru"] = client };

            var imageBytesByUrl = posts.ToDictionary(p => p.FileUrl, p => MakeDistinctTestPng((int)p.PostId));
            using var downloadClient = new HttpClient(new RoutedImageHttpMessageHandler(imageBytesByUrl));

            await DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize,
                minImages: 1, maxImages: maxImages, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress: null, CancellationToken.None);

            var progress = await db.GetTagProgressAsync("1girl", "danbooru", "positive");
            Assert.False(progress.Done); // pagination genuinely NOT exhausted -- only quota was met
            Assert.Equal(maxImages, progress.QuotaSatisfiedAtMaxImages);

            // The page-boundary checkpoint (not just the quota flag) must reflect a real,
            // correctly-bounded count -- at least enough to have satisfied the floor, but
            // never more than the posts actually available to dispatch.
            Assert.InRange(progress.SitePositiveCount, maxImages, posts.Count);

            var callsAfterFirstRun = client.ListPostsCallCount;
            Assert.True(callsAfterFirstRun > 0);

            // Second run: same db, same directory, same maxImages, same client instance
            // (so its own call counter carries over) -- if the fast path weren't
            // working, this would fetch at least one more page.
            await DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize,
                minImages: 1, maxImages: maxImages, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress: null, CancellationToken.None);

            Assert.Equal(callsAfterFirstRun, client.ListPostsCallCount);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task RunAsync_ResumedWithHigherMaxImagesThanRecordedQuota_ResumesFetching()
    {
        const int inputSize = 8;
        const int firstRunMaxImages = 3;
        const int secondRunMaxImages = firstRunMaxImages * 3;

        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            await db.UpsertTagSurveysAsync([("1girl", (int?)1000, (int?)null, true)], DateTimeOffset.UtcNow, null);

            // First page satisfies firstRunMaxImages without exhausting pagination (same
            // setup as RunAsync_ResumedWithQuotaAlreadySatisfied_SkipsRefetchingEntirely);
            // a second page then lets the run continue once a HIGHER --max-images makes
            // the previously-recorded quota untrustworthy.
            var page1Posts = Enumerable.Range(0, firstRunMaxImages + 2)
                .Select(i => new BooruPost(i, $"img-{i}", new Uri($"https://example.test/page1-{i}.png"), ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8))
                .ToList();
            var page2Posts = Enumerable.Range(0, secondRunMaxImages)
                .Select(i => new BooruPost(1000 + i, $"img2-{i}", new Uri($"https://example.test/page2-{i}.png"), ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8))
                .ToList();
            var client = new CountingCursorPagedSiteClient("danbooru",
                firstPage: new BooruPostPage(page1Posts, "page-2-cursor"),
                pagesByCursor: new Dictionary<string, BooruPostPage> { ["page-2-cursor"] = new BooruPostPage(page2Posts, null) });
            var clients = new Dictionary<string, IBooruClient> { ["danbooru"] = client };

            var imageBytesByUrl = page1Posts.Concat(page2Posts).ToDictionary(p => p.FileUrl, p => MakeDistinctTestPng((int)p.PostId));
            using var downloadClient = new HttpClient(new RoutedImageHttpMessageHandler(imageBytesByUrl));

            await DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize,
                minImages: 1, maxImages: firstRunMaxImages, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress: null, CancellationToken.None);

            var afterFirstRun = await db.GetTagProgressAsync("1girl", "danbooru", "positive");
            Assert.Equal(firstRunMaxImages, afterFirstRun.QuotaSatisfiedAtMaxImages);
            var callsAfterFirstRun = client.ListPostsCallCount;

            // A higher --max-images can't trust the previously-recorded quota (a smaller
            // quota being met doesn't mean a larger one is) -- this must fall back to the
            // normal path and fetch page 2.
            await DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize,
                minImages: 1, maxImages: secondRunMaxImages, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress: null, CancellationToken.None);

            Assert.True(client.ListPostsCallCount > callsAfterFirstRun);
            Assert.True((await db.GetAllImagesAsync()).Count > firstRunMaxImages);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static void AssertNeverRegressesExceptOnReset(IReadOnlyList<long> values)
    {
        for (var i = 1; i < values.Count; i++)
        {
            if (values[i] == 0)
                continue; // a fresh page's reset, not a regression

            Assert.True(values[i] >= values[i - 1],
                $"Progress regressed from {values[i - 1]} to {values[i]} at index {i} — a stale concurrent report overwrote a fresher one.");
        }
    }
}
