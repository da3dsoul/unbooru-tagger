using System.Net;
using System.Security.Cryptography;
using SkiaSharp;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

public class DatasetRecoveryTests
{
    private static async Task<CrawlDatabase> OpenTempDatabaseAsync(string dir)
    {
        Directory.CreateDirectory(dir);
        return await CrawlDatabase.OpenOrCreateAsync(dir);
    }

    private static byte[] MakePng(byte r, byte g, byte b)
    {
        using var bitmap = new SKBitmap(new SKImageInfo(8, 8, SKColorType.Rgba8888, SKAlphaType.Unpremul));
        using (var canvas = new SKCanvas(bitmap))
            canvas.Clear(new SKColor(r, g, b, 255));
        using var encoded = bitmap.Encode(SKEncodedImageFormat.Png, 100);
        return encoded.ToArray();
    }

    private static string Md5Hex(byte[] bytes) => Convert.ToHexStringLower(MD5.HashData(bytes));

    [Fact]
    public async Task ComputePlanAsync_OrdersByCacheRowIndexAndExcludesSoleUnsupportedMediaSources()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"recovery-plan-{Guid.NewGuid():N}");
        using var db = await OpenTempDatabaseAsync(dir);
        try
        {
            // md5-gif: only ever sourced from a .gif URL -- must be excluded entirely.
            // md5-mixed: one real source AND a .gif source -- kept, but the gif row flagged to strip.
            // md5-real: a normal image, one real source.
            await db.CommitPendingImagesAsync(
                [
                    new PendingNewImage("md5-real", 0, 100, 100, DateTimeOffset.UtcNow, "danbooru", 1, "https://example/a.jpg", "g", DateTimeOffset.UtcNow, ["1girl"], PHash: 0),
                    new PendingNewImage("md5-gif", 1, 100, 100, DateTimeOffset.UtcNow, "danbooru", 2, "https://example/b.gif", "g", DateTimeOffset.UtcNow, ["1girl"], PHash: 0),
                    new PendingNewImage("md5-mixed", 2, 100, 100, DateTimeOffset.UtcNow, "danbooru", 3, "https://example/c.jpg", "g", DateTimeOffset.UtcNow, ["1girl"], PHash: 0),
                ],
                [
                    new PendingAdditionalSource("md5-mixed", "gelbooru", 4, "https://example/c.gif", "g", DateTimeOffset.UtcNow, ["1girl"], DateTimeOffset.UtcNow),
                ],
                [], CancellationToken.None);

            var plan = await DatasetRecovery.ComputePlanAsync(db);

            Assert.Equal(["md5-real", "md5-mixed"], plan.OrderedMd5s);
            Assert.Equal(["md5-gif"], plan.SoleSourceMd5s);
            Assert.Single(plan.ExtraSourcesToStrip);
            Assert.Equal(("md5-mixed", "gelbooru", 4L), plan.ExtraSourcesToStrip[0]);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    private static void WriteTagRowSource(string directory, int inputSize, int rowCount)
    {
        using var writer = new PreprocessedDatasetCacheWriter(directory, inputSize);
        for (var i = 0; i < rowCount; i++)
            writer.Append(new EncodedImage(new byte[inputSize * inputSize * 3], new LetterboxBox(0, 0, inputSize, inputSize)), [10 + i]);
    }

    [Fact]
    public async Task FillRemainingAsync_DownloadsVerifiesMd5FallsBackOnMismatchAndLogsPermanentFailures()
    {
        const int inputSize = 8;
        var stagingDir = Path.Combine(Path.GetTempPath(), $"recovery-staging-{Guid.NewGuid():N}");
        var tagRowSourceDir = Path.Combine(Path.GetTempPath(), $"recovery-tagsrc-{Guid.NewGuid():N}");
        try
        {
            WriteTagRowSource(tagRowSourceDir, inputSize, 3);

            var pngA = MakePng(200, 0, 0);
            var pngBGood = MakePng(0, 200, 0);
            var pngBBad = MakePng(0, 0, 200); // served at md5-b's FIRST candidate URL -- wrong content

            var md5A = Md5Hex(pngA);
            var md5B = Md5Hex(pngBGood);
            const string md5C = "c-never-recoverable";

            var urlA = new Uri("https://example/a.jpg");
            var urlBBad = new Uri("https://example/b-bad.jpg");
            var urlBGood = new Uri("https://example/b-good.jpg");
            var urlCMissing = new Uri("https://example/c-missing.jpg");

            var imageBytesByUrl = new Dictionary<Uri, byte[]> { [urlA] = pngA, [urlBBad] = pngBBad, [urlBGood] = pngBGood };
            var statusByUrl = new Dictionary<Uri, HttpStatusCode> { [urlCMissing] = HttpStatusCode.NotFound };
            using var httpClient = new HttpClient(new RoutedStatusHttpMessageHandler(imageBytesByUrl, statusByUrl));

            var orderedMd5s = new List<string> { md5A, md5B, md5C };
            var sourceUrls = new Dictionary<string, IReadOnlyList<(string Site, string PostUrl)>>
            {
                [md5A] = [("danbooru", urlA.ToString())],
                [md5B] = [("danbooru", urlBBad.ToString()), ("danbooru", urlBGood.ToString())],
                [md5C] = [("danbooru", urlCMissing.ToString())],
            };
            var rateLimiters = new Dictionary<string, IRateLimiter> { ["danbooru"] = new FixedIntervalRateLimiter(1000) };
            var rates = new Dictionary<string, double> { ["danbooru"] = 1000 };

            var progressCalls = new List<(int, int)>();
            var result = await DatasetRecovery.FillRemainingAsync(
                stagingDir, tagRowSourceDir, inputSize, orderedMd5s, sourceUrls, httpClient, rateLimiters, rates, concurrency: 2,
                onProgress: (done, total) => progressCalls.Add((done, total)));

            Assert.Equal([md5A, md5B], result.WrittenMd5s);
            Assert.Equal([md5C], result.PermanentlyFailedMd5s);

            using var reader = new PreprocessedDatasetCacheReader(stagingDir);
            Assert.Equal(2, reader.ImageCount);
            Assert.Equal([10], reader.ImageTagRows[0]); // row 0's tag rows, reused from tagRowSourceDir despite the pixels being freshly downloaded
            Assert.Equal([11], reader.ImageTagRows[1]);

            var failureLines = File.ReadAllLines(Path.Combine(stagingDir, DatasetRecovery.FailuresLogFileName));
            Assert.Single(failureLines);
            Assert.StartsWith(md5C, failureLines[0]);

            Assert.Equal((3, 3), progressCalls[^1]);
        }
        finally
        {
            if (Directory.Exists(stagingDir))
                Directory.Delete(stagingDir, recursive: true);
            Directory.Delete(tagRowSourceDir, recursive: true);
        }
    }

    [Fact]
    public async Task FillRemainingAsync_ResumesFromWhereAnInterruptedRunLeftOff()
    {
        const int inputSize = 8;
        var stagingDir = Path.Combine(Path.GetTempPath(), $"recovery-resume-staging-{Guid.NewGuid():N}");
        var tagRowSourceDir = Path.Combine(Path.GetTempPath(), $"recovery-resume-tagsrc-{Guid.NewGuid():N}");
        try
        {
            WriteTagRowSource(tagRowSourceDir, inputSize, 3);

            var pngs = Enumerable.Range(0, 3).Select(i => MakePng((byte)(50 * i), 0, 0)).ToList();
            var md5s = pngs.Select(Md5Hex).ToList();
            var urls = Enumerable.Range(0, 3).Select(i => new Uri($"https://example/{i}.jpg")).ToList();

            var imageBytesByUrl = Enumerable.Range(0, 3).ToDictionary(i => urls[i], i => pngs[i]);
            using var httpClient = new HttpClient(new RoutedStatusHttpMessageHandler(imageBytesByUrl, new Dictionary<Uri, HttpStatusCode>()));

            var sourceUrls = Enumerable.Range(0, 3)
                .ToDictionary(i => md5s[i], i => (IReadOnlyList<(string Site, string PostUrl)>)[("danbooru", urls[i].ToString())]);
            var rateLimiters = new Dictionary<string, IRateLimiter> { ["danbooru"] = new FixedIntervalRateLimiter(1000) };
            var rates = new Dictionary<string, double> { ["danbooru"] = 1000 };

            using (var cts = new CancellationTokenSource())
            {
                var reportedTotals = new List<int>();
                await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
                    DatasetRecovery.FillRemainingAsync(
                        stagingDir, tagRowSourceDir, inputSize, md5s, sourceUrls, httpClient, rateLimiters, rates, concurrency: 1,
                        onProgress: (done, _) =>
                        {
                            reportedTotals.Add(done);
                            if (done == 1)
                                cts.Cancel();
                        },
                        cts.Token));
            }

            using (var reader = new PreprocessedDatasetCacheReader(stagingDir))
                Assert.Equal(1, reader.ImageCount); // exactly the one batch that completed before cancellation

            var result = await DatasetRecovery.FillRemainingAsync(
                stagingDir, tagRowSourceDir, inputSize, md5s, sourceUrls, httpClient, rateLimiters, rates, concurrency: 1);

            Assert.Equal(md5s, result.WrittenMd5s);
            Assert.Empty(result.PermanentlyFailedMd5s);

            using var finalReader = new PreprocessedDatasetCacheReader(stagingDir);
            Assert.Equal(3, finalReader.ImageCount);
        }
        finally
        {
            if (Directory.Exists(stagingDir))
                Directory.Delete(stagingDir, recursive: true);
            Directory.Delete(tagRowSourceDir, recursive: true);
        }
    }

    /// <summary>Records which URL was requested for each image, in request order -- for asserting the site round-robin's actual, observed dispatch pattern rather than trusting it worked from the outside.</summary>
    private sealed class RecordingHttpMessageHandler(IReadOnlyDictionary<Uri, byte[]> imageBytesByUrl) : HttpMessageHandler
    {
        private readonly List<Uri> _requested = [];
        public IReadOnlyList<Uri> Requested => _requested;

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            lock (_requested)
                _requested.Add(request.RequestUri!);
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(imageBytesByUrl[request.RequestUri!]) });
        }
    }

    /// <summary>
    /// Regression test for a real near-miss: a first attempt at multi-site scheduling
    /// unconditionally preferred Danbooru whenever an image had a source on both sites,
    /// which for a corpus where most images DO have a Danbooru source meant Gelbooru's
    /// own rate-limited capacity sat completely idle -- the whole re-download phase's
    /// throughput was effectively capped at Danbooru's rate alone, not the two
    /// combined. This asserts the fix actually interleaves proportionally to each
    /// site's configured rate instead of exhausting one site first.
    /// </summary>
    [Fact]
    public async Task FillRemainingAsync_BalancesBothSitesProportionallyToTheirRatesWhenBothHaveASource()
    {
        const int inputSize = 8;
        var stagingDir = Path.Combine(Path.GetTempPath(), $"recovery-balance-staging-{Guid.NewGuid():N}");
        var tagRowSourceDir = Path.Combine(Path.GetTempPath(), $"recovery-balance-tagsrc-{Guid.NewGuid():N}");
        try
        {
            const int imageCount = 30;
            WriteTagRowSource(tagRowSourceDir, inputSize, imageCount);

            var pngs = Enumerable.Range(0, imageCount).Select(i => MakePng((byte)(i * 7 % 256), 0, 0)).ToList();
            var md5s = pngs.Select(Md5Hex).ToList();

            var imageBytesByUrl = new Dictionary<Uri, byte[]>();
            var sourceUrls = new Dictionary<string, IReadOnlyList<(string Site, string PostUrl)>>();
            for (var i = 0; i < imageCount; i++)
            {
                var danbooruUrl = new Uri($"https://danbooru.example/{i}.jpg");
                var gelbooruUrl = new Uri($"https://gelbooru.example/{i}.jpg");
                imageBytesByUrl[danbooruUrl] = pngs[i];
                imageBytesByUrl[gelbooruUrl] = pngs[i]; // identical content on both sites -- either satisfies the MD5 check
                sourceUrls[md5s[i]] = [("danbooru", danbooruUrl.ToString()), ("gelbooru", gelbooruUrl.ToString())];
            }

            var handler = new RecordingHttpMessageHandler(imageBytesByUrl);
            using var httpClient = new HttpClient(handler);
            var rateLimiters = new Dictionary<string, IRateLimiter> { ["danbooru"] = new FixedIntervalRateLimiter(1000), ["gelbooru"] = new FixedIntervalRateLimiter(1000) };
            var rates = new Dictionary<string, double> { ["danbooru"] = 4, ["gelbooru"] = 2 }; // this project's own defaults

            var result = await DatasetRecovery.FillRemainingAsync(
                stagingDir, tagRowSourceDir, inputSize, md5s, sourceUrls, httpClient, rateLimiters, rates, concurrency: 1);

            Assert.Equal(imageCount, result.WrittenMd5s.Count);
            Assert.Empty(result.PermanentlyFailedMd5s);

            // concurrency: 1 makes dispatch order deterministic (one image fully resolved before the next launches) for this assertion.
            var sitesInOrder = handler.Requested.Select(u => u.Host.StartsWith("danbooru") ? "danbooru" : "gelbooru").ToList();
            Assert.Equal(imageCount, sitesInOrder.Count);

            var danbooruCount = sitesInOrder.Count(s => s == "danbooru");
            var gelbooruCount = sitesInOrder.Count(s => s == "gelbooru");
            // Roughly 4:2 -- smooth WRR over 30 picks lands close to but not necessarily exactly 20:10.
            Assert.InRange(danbooruCount, 18, 22);
            Assert.InRange(gelbooruCount, 8, 12);

            // Interleaved, not clustered: no run of the same site longer than 3 in a row
            // (a fixed "always prefer danbooru" bug would produce one long danbooru run).
            var longestRun = 1;
            var currentRun = 1;
            for (var i = 1; i < sitesInOrder.Count; i++)
            {
                currentRun = sitesInOrder[i] == sitesInOrder[i - 1] ? currentRun + 1 : 1;
                longestRun = Math.Max(longestRun, currentRun);
            }
            Assert.True(longestRun <= 3, $"Longest same-site run was {longestRun}, expected interleaving to keep it short.");
        }
        finally
        {
            if (Directory.Exists(stagingDir))
                Directory.Delete(stagingDir, recursive: true);
            Directory.Delete(tagRowSourceDir, recursive: true);
        }
    }
}
