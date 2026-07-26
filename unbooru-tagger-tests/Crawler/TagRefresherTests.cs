using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;
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

/// <summary>Like <see cref="FakeRefreshClient"/> but records every post id it was actually asked for — needed to prove a targeted refresh never touches sources outside its scope, not just that it produces the right answer for the ones it does touch.</summary>
internal sealed class RecordingRefreshClient(string siteName, Dictionary<long, BooruPost?> postsById) : IBooruClient
{
    public string SiteName => siteName;
    public int PageSize => 100;
    public IRateLimiter RateLimiter { get; } = new ImmediateRateLimiter();
    public HashSet<long> RequestedPostIds { get; } = [];

    public IAsyncEnumerable<BooruTagCount> ListTagsByCountDescendingAsync(CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<BooruPostPage> ListPostsAsync(string tagQuery, string? cursor, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<BooruPost?> GetPostAsync(long postId, CancellationToken cancellationToken = default)
    {
        RequestedPostIds.Add(postId);
        return Task.FromResult(postsById.GetValueOrDefault(postId));
    }
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
            writer.Append(new EncodedImage(Enumerable.Range(0, 12).Select(i => (byte)i).ToArray(), new LetterboxBox(0, 0, 2, 2)), [0]);

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
    public async Task RunAsync_ToleratesLegacyDuplicateRawNameSurveyRows_WithoutCrashing()
    {
        var (directory, db) = await SeedDatasetAsync();
        try
        {
            // Simulates a crawl.sqlite surveyed before TagSurveyor started merging by raw
            // name: two sites categorized the same raw tag differently, leaving both
            // "elvaan" (General, danbooru) and "character:elvaan" (Character, gelbooru) as
            // separate eligible rows. RunAsync must not crash building its raw-name ->
            // identity lookup (TagRowMutations.BuildEligibleIdentities), and should
            // deterministically pick the identity with the higher combined post count.
            await db.UpsertTagSurveysAsync(
                [("elvaan", (int?)1000, null, true), ("character:elvaan", null, (int?)2000, true)],
                DateTimeOffset.UtcNow, null);

            var clients = new Dictionary<string, IBooruClient>
            {
                ["gelbooru"] = new FakeRefreshClient("gelbooru", new Dictionary<long, BooruPost?> { [2] = MakePost(2, "1girl", "elvaan") }),
            };

            var result = await TagRefresher.RunAsync(db, clients, directory, inputSize: 2, minImages: 1, reset: false, progress: null, CancellationToken.None);

            Assert.Equal(1, result.ImagesChanged);

            var vocabulary = TagVocabulary.Load(Path.Combine(directory, "tag_vocabulary.json"), Path.Combine(directory, "tag_vocabulary.delta.jsonl"));
            var tagNames = ReadRow0Tags(directory).Select(idx => vocabulary.GetByRowIndex(idx).Tag).ToHashSet();
            Assert.Contains("character:elvaan", tagNames);
            Assert.DoesNotContain("elvaan", tagNames);
        }
        finally
        {
            db.Dispose();
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task RunAsync_ReconcilesAnAliasedAwayTag_ToItsMergedIdentity_WhenTagAliasesProvided()
    {
        // Simulates the exact corruption a pre-fix crawl left behind: an image tagged
        // "head_pat" (its own vocabulary row) back when that was still its own eligible
        // survey tag, before TagSurveyor merged it into "headpat". A plain refresh-tags
        // run (no tagAliases) can't fix this — its single source, Gelbooru, will forever
        // keep reporting the raw string "head_pat" verbatim, which without alias
        // knowledge resolves to nothing at all (see TagRowMutations.BuildEligibleIdentities).
        // With tagAliases supplied, that same fresh "head_pat" observation must resolve
        // to "headpat", reconciling the image onto the merged identity.
        var directory = Directory.CreateTempSubdirectory().FullName;

        var vocabulary = TagVocabulary.CreateEmpty();
        vocabulary.AddTag("head_pat"); // row 0 — the pre-merge, now-orphaned identity
        vocabulary.Save(Path.Combine(directory, "tag_vocabulary.json"));

        using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize: 2))
            writer.Append(new EncodedImage(Enumerable.Range(0, 12).Select(i => (byte)i).ToArray(), new LetterboxBox(0, 0, 2, 2)), [0]);

        var db = await CrawlDatabase.OpenOrCreateAsync(directory);
        try
        {
            // Post-merge survey state: only "headpat" is its own eligible tag now.
            await db.UpsertTagSurveysAsync([("headpat", (int?)1000, (int?)1000, true)], DateTimeOffset.UtcNow, null);

            var image = new PendingNewImage("headpat-img", 0, 100, 100, DateTimeOffset.UtcNow, "gelbooru", 2, "https://x/2.jpg", "g", DateTimeOffset.UtcNow, ["head_pat"], PHash: 0);
            await db.CommitPendingImagesAsync([image], [], [], CancellationToken.None);

            var clients = new Dictionary<string, IBooruClient>
            {
                ["gelbooru"] = new FakeRefreshClient("gelbooru", new Dictionary<long, BooruPost?> { [2] = MakePost(2, "head_pat") }),
            };
            var tagAliases = new Dictionary<string, string> { ["head_pat"] = "headpat" };

            var result = await TagRefresher.RunAsync(db, clients, directory, inputSize: 2, minImages: 1, reset: false, progress: null, CancellationToken.None, tagAliases: tagAliases);

            Assert.Equal(1, result.ImagesChanged);

            var reloadedVocabulary = TagVocabulary.Load(Path.Combine(directory, "tag_vocabulary.json"), Path.Combine(directory, "tag_vocabulary.delta.jsonl"));
            var tagNames = ReadRow0Tags(directory).Select(idx => reloadedVocabulary.GetByRowIndex(idx).Tag).ToHashSet();
            Assert.Equal(["headpat"], tagNames); // reconciled onto the merged identity, old one dropped
        }
        finally
        {
            db.Dispose();
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task RunAsync_OnlyTagsAffectingImages_ChecksOnlyImagesHoldingThatTag_LeavingOthersAndTheRealCursorUntouched()
    {
        // Two unrelated images: one holds "head_pat" (targeted), one only holds "1girl"
        // (must never even be fetched) — proves a targeted pass resolves its own working
        // set from the vocabulary instead of falling back to a full sweep.
        var directory = Directory.CreateTempSubdirectory().FullName;

        var vocabulary = TagVocabulary.CreateEmpty();
        vocabulary.AddTag("head_pat"); // row 0
        vocabulary.AddTag("1girl");    // row 1
        vocabulary.Save(Path.Combine(directory, "tag_vocabulary.json"));

        using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize: 2))
        {
            writer.Append(new EncodedImage(Enumerable.Range(0, 12).Select(i => (byte)i).ToArray(), new LetterboxBox(0, 0, 2, 2)), [0]); // row 0 image: head_pat
            writer.Append(new EncodedImage(Enumerable.Range(0, 12).Select(i => (byte)i).ToArray(), new LetterboxBox(0, 0, 2, 2)), [1]); // row 1 image: 1girl
        }

        var db = await CrawlDatabase.OpenOrCreateAsync(directory);
        try
        {
            await db.UpsertTagSurveysAsync([("headpat", (int?)1000, (int?)1000, true)], DateTimeOffset.UtcNow, null);

            var targetedImage = new PendingNewImage("target-img", 0, 100, 100, DateTimeOffset.UtcNow, "gelbooru", 10, "https://x/10.jpg", "g", DateTimeOffset.UtcNow, ["head_pat"], PHash: 0);
            var untouchedImage = new PendingNewImage("untouched-img", 1, 100, 100, DateTimeOffset.UtcNow, "gelbooru", 20, "https://x/20.jpg", "g", DateTimeOffset.UtcNow, ["1girl"], PHash: 0);
            await db.CommitPendingImagesAsync([targetedImage, untouchedImage], [], [], CancellationToken.None);

            var client = new RecordingRefreshClient("gelbooru", new Dictionary<long, BooruPost?> { [10] = MakePost(10, "head_pat") });
            var clients = new Dictionary<string, IBooruClient> { ["gelbooru"] = client };
            var tagAliases = new Dictionary<string, string> { ["head_pat"] = "headpat" };

            var result = await TagRefresher.RunAsync(
                db, clients, directory, inputSize: 2, minImages: 1, reset: false, progress: null, CancellationToken.None,
                excludedTags: null, tagAliases: tagAliases, onlyTagsAffectingImages: ["head_pat"]);

            Assert.Equal(1, result.SourcesChecked);
            Assert.Contains(10L, client.RequestedPostIds);
            Assert.DoesNotContain(20L, client.RequestedPostIds); // the untouched image's source was never even requested

            var reloadedVocabulary = TagVocabulary.Load(Path.Combine(directory, "tag_vocabulary.json"), Path.Combine(directory, "tag_vocabulary.delta.jsonl"));
            using var reader = PreprocessedDatasetCacheWriter.OpenOrCreate(directory, inputSize: 2);
            var committedTagRows = reader.ReadCommittedTagRows();
            var row0Tags = committedTagRows[0].Select(idx => reloadedVocabulary.GetByRowIndex(idx).Tag).ToHashSet();
            var row1Tags = committedTagRows[1].Select(idx => reloadedVocabulary.GetByRowIndex(idx).Tag).ToHashSet();
            Assert.Equal(["headpat"], row0Tags); // reconciled onto the merged identity
            Assert.Equal(["1girl"], row1Tags);   // completely unchanged

            // A targeted pass must never advance (or create a misleading) real cursor —
            // gelbooru was never swept normally, so its RefreshProgress must still read
            // as untouched afterward.
            var (lastPostId, done) = await db.GetRefreshProgressAsync("gelbooru");
            Assert.Equal(0, lastPostId);
            Assert.False(done);
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
