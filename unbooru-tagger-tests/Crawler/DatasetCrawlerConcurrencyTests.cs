using System.Net;
using SkiaSharp;
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

/// <summary>Answers every request with the same image bytes after an artificial delay — long enough that two concurrent site workers' downloads are reliably still in flight at the same time, widening the race window <see cref="DatasetCrawler.ProcessPostAsync"/>'s double-checked locking has to handle correctly.</summary>
internal sealed class DelayedImageHttpMessageHandler(byte[] imageBytes, TimeSpan delay) : HttpMessageHandler
{
    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
        return new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(imageBytes) };
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
    /// recover the gap's provenance. Here a page of two posts is processed with the
    /// second post's download cancelled mid-flight (simulating an exit request arriving
    /// between them) — the first post must still be durably committed to crawl.sqlite,
    /// not just sitting in the cache file, once the run finishes unwinding.
    /// </summary>
    [Fact]
    public async Task RunAsync_CancelledMidPage_StillDurablyCommitsPostsAlreadyAppended()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            await db.UpsertTagSurveysAsync([("1girl", (int?)10, (int?)null, true)], DateTimeOffset.UtcNow, null);

            var post1 = new BooruPost(1, "already-appended-before-cancel", new Uri("https://example.test/1.png"), ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8);
            var post2 = new BooruPost(2, "never-reached-cancelled-mid-download", new Uri("https://example.test/2.png"), ["1girl"], "g", DateTimeOffset.UtcNow, 8, 8);

            var clients = new Dictionary<string, IBooruClient> { ["danbooru"] = new MultiPostSiteClient("danbooru", [post1, post2]) };

            using var cts = new CancellationTokenSource();
            // Cancels on the 2nd HTTP request (post2's download) — post1's download (the
            // 1st request) must already have succeeded and been appended by then.
            using var downloadClient = new HttpClient(new CancelOnRequestHttpMessageHandler(MakeTestPng(), cancelOnRequestNumber: 2, cts));

            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => DatasetCrawler.RunAsync(
                db, clients, downloadClient, directory, inputSize: 8,
                minImages: 1, maxImages: 3, negativeTarget: 0, vocabCompactIntervalPages: 1,
                progress: null, cts.Token));

            var images = await db.GetAllImagesAsync();
            Assert.Single(images); // post1 must have made it into crawl.sqlite, not just the cache file
            Assert.Equal("already-appended-before-cancel", images[0].Md5);
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
}
