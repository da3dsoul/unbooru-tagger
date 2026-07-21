using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

/// <summary>Answers <see cref="IBooruClient.GetPostAsync"/> from an in-memory map — everything else <see cref="TagRefresher"/> doesn't call is unsupported on purpose, so a test that accidentally exercises it fails loudly instead of silently returning nonsense.</summary>
internal sealed class FakeRefreshClient(string siteName, Dictionary<long, BooruPost?> postsById) : IBooruClient
{
    public string SiteName => siteName;
    public int PageSize => 100;
    public IRateLimiter RateLimiter { get; } = new ImmediateRateLimiter();

    public IAsyncEnumerable<BooruTagCount> ListTagsByCountDescendingAsync(CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<BooruPostPage> ListPostsAsync(string tagQuery, string? cursor, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<BooruPost?> GetPostAsync(long postId, CancellationToken cancellationToken = default) =>
        Task.FromResult(postsById.GetValueOrDefault(postId));
}

public class TagRefresherTests
{
    private static BooruPost MakePost(long id, params string[] tags) =>
        new(id, "unused", new Uri("https://example/x.jpg"), tags, "g", DateTimeOffset.UtcNow, 100, 100);

    /// <summary>Seeds a dataset directory + crawl.sqlite with one canonical image (row 0, tag "1girl") sourced from danbooru post 1 and gelbooru post 2, both already carrying "1girl" as their captured snapshot — the state a normal crawl would have left behind.</summary>
    private static async Task<(string Directory, CrawlDatabase Db)> SeedDatasetAsync()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;

        var vocabulary = TagVocabulary.CreateEmpty();
        vocabulary.AddTag("1girl"); // row 0
        vocabulary.Save(Path.Combine(directory, "tag_vocabulary.json"));

        using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize: 2))
            writer.Append(Enumerable.Range(0, 12).Select(i => (float)i).ToArray(), [0]);

        var db = await CrawlDatabase.OpenOrCreateAsync(directory);
        await db.UpsertTagSurveysAsync(
            [("1girl", (int?)1000, (int?)1000, true), ("outdoors", (int?)1000, (int?)1000, true)],
            DateTimeOffset.UtcNow, null);

        var image = new PendingNewImage("abc", 0, 100, 100, DateTimeOffset.UtcNow, "danbooru", 1, "https://x/1.jpg", "g", DateTimeOffset.UtcNow, ["1girl"], PHash: 0);
        await db.CommitPendingImagesAsync([image], [], [], CancellationToken.None);
        var additional = new PendingAdditionalSource("abc", "gelbooru", 2, "https://x/2.jpg", "g", DateTimeOffset.UtcNow, ["1girl"], DateTimeOffset.UtcNow);
        await db.CommitPendingImagesAsync([], [additional], [], CancellationToken.None);

        return (directory, db);
    }

    private static IReadOnlyList<int> ReadRow0Tags(string directory)
    {
        using var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(directory, inputSize: 2);
        return writer.ReadCommittedTagRows()[0];
    }

    [Fact]
    public async Task RunAsync_AddsATagFromARefreshedSource_WithoutRemovingATagAKnownSiblingStillAsserts()
    {
        var (directory, db) = await SeedDatasetAsync();
        try
        {
            // Only gelbooru gets refreshed, and it no longer shows "1girl" — but
            // danbooru's post 1 was never refreshed this run, so its last-captured
            // snapshot (["1girl"], from SeedDatasetAsync) is still "known", not
            // "unknown" — reconciliation must not drop "1girl" on that basis alone.
            var clients = new Dictionary<string, IBooruClient>
            {
                ["gelbooru"] = new FakeRefreshClient("gelbooru", new Dictionary<long, BooruPost?> { [2] = MakePost(2, "outdoors") }),
            };

            var result = await TagRefresher.RunAsync(db, clients, directory, inputSize: 2, minImages: 1, reset: false, progress: null, CancellationToken.None);

            Assert.Equal(1, result.SourcesChecked);
            Assert.Equal(1, result.ImagesChanged);
            Assert.Empty(result.FailedSites);

            var vocabulary = TagVocabulary.Load(Path.Combine(directory, "tag_vocabulary.json"), Path.Combine(directory, "tag_vocabulary.delta.jsonl"));
            var tagNames = ReadRow0Tags(directory).Select(idx => vocabulary.GetByRowIndex(idx).Tag).ToHashSet();
            Assert.Equal(["1girl", "outdoors"], tagNames.Order());

            var counts = await db.GetAllCombinedPositiveCountsAsync();
            Assert.Equal(1, counts["1girl"]);
            Assert.Equal(1, counts["outdoors"]);
        }
        finally
        {
            db.Dispose();
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task RunAsync_RemovesATag_OnceEveryKnownSourceNoLongerAssertsIt()
    {
        var (directory, db) = await SeedDatasetAsync();
        try
        {
            // Both sites get refreshed and neither shows "1girl" anymore — only once
            // every known source has actually confirmed its absence should it drop.
            var clients = new Dictionary<string, IBooruClient>
            {
                ["danbooru"] = new FakeRefreshClient("danbooru", new Dictionary<long, BooruPost?> { [1] = MakePost(1) }),
                ["gelbooru"] = new FakeRefreshClient("gelbooru", new Dictionary<long, BooruPost?> { [2] = MakePost(2, "outdoors") }),
            };

            var result = await TagRefresher.RunAsync(db, clients, directory, inputSize: 2, minImages: 1, reset: false, progress: null, CancellationToken.None);

            Assert.Equal(2, result.SourcesChecked);

            var vocabulary = TagVocabulary.Load(Path.Combine(directory, "tag_vocabulary.json"), Path.Combine(directory, "tag_vocabulary.delta.jsonl"));
            var tagNames = ReadRow0Tags(directory).Select(idx => vocabulary.GetByRowIndex(idx).Tag).ToHashSet();
            Assert.Equal(["outdoors"], tagNames);

            var counts = await db.GetAllCombinedPositiveCountsAsync();
            Assert.Equal(0, counts["1girl"]);
            Assert.Equal(1, counts["outdoors"]);
        }
        finally
        {
            db.Dispose();
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task RunAsync_DeletedPost_IsTreatedAsAssertingNoTags_NotAsUnknown()
    {
        var (directory, db) = await SeedDatasetAsync();
        try
        {
            var clients = new Dictionary<string, IBooruClient>
            {
                ["danbooru"] = new FakeRefreshClient("danbooru", new Dictionary<long, BooruPost?> { [1] = null }),
                ["gelbooru"] = new FakeRefreshClient("gelbooru", new Dictionary<long, BooruPost?> { [2] = null }),
            };

            var result = await TagRefresher.RunAsync(db, clients, directory, inputSize: 2, minImages: 1, reset: false, progress: null, CancellationToken.None);

            Assert.Equal(1, result.ImagesChanged);
            Assert.Empty(ReadRow0Tags(directory));

            // A deleted post is a *confirmed* zero-tags state (Tags = "[]"), not the
            // never-captured "unknown" state (Tags = NULL) — that distinction is exactly
            // what let removal trigger at all once both sources confirmed it's gone.
            var snapshots = await db.GetImageSourceSnapshotsAsync("abc");
            Assert.All(snapshots, s => Assert.NotNull(s.Tags));
            Assert.All(snapshots, s => Assert.Empty(s.Tags!));
        }
        finally
        {
            db.Dispose();
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task RunAsync_IsResumable_SecondRunOnlyChecksSourcesPastTheSavedCursor()
    {
        var (directory, db) = await SeedDatasetAsync();
        try
        {
            var clients1 = new Dictionary<string, IBooruClient>
            {
                ["danbooru"] = new FakeRefreshClient("danbooru", new Dictionary<long, BooruPost?> { [1] = MakePost(1, "1girl") }),
            };
            await TagRefresher.RunAsync(db, clients1, directory, inputSize: 2, minImages: 1, reset: false, progress: null, CancellationToken.None);

            var (lastPostId, done) = await db.GetRefreshProgressAsync("danbooru");
            Assert.Equal(1, lastPostId);
            Assert.True(done);

            // A second run with no new sources since the first must find nothing to do —
            // GetSourcesBatchAsync(after: 1) is empty, so GetPostAsync is never called.
            var clients2 = new Dictionary<string, IBooruClient>
            {
                ["danbooru"] = new FakeRefreshClient("danbooru", new Dictionary<long, BooruPost?>()), // any lookup here would return null and fail the run's assumptions
            };
            var result2 = await TagRefresher.RunAsync(db, clients2, directory, inputSize: 2, minImages: 1, reset: false, progress: null, CancellationToken.None);

            Assert.Equal(0, result2.SourcesChecked);
        }
        finally
        {
            db.Dispose();
            Directory.Delete(directory, recursive: true);
        }
    }
}
