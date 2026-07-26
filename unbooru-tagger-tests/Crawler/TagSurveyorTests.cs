using System.Runtime.CompilerServices;
using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

/// <summary>Answers <see cref="IBooruClient.ListTagsByCountDescendingAsync"/> from a fixed, already-sorted list — everything else <see cref="TagSurveyor"/> doesn't call is unsupported on purpose.</summary>
internal sealed class FakeSurveyClient(string siteName, IReadOnlyList<BooruTagCount> tags) : IBooruClient
{
    public string SiteName => siteName;
    public int PageSize => 100;
    public IRateLimiter RateLimiter { get; } = new ImmediateRateLimiter();

    public async IAsyncEnumerable<BooruTagCount> ListTagsByCountDescendingAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foreach (var tag in tags)
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return tag;
            await Task.Yield();
        }
    }

    public Task<BooruPostPage> ListPostsAsync(string tagQuery, string? cursor, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<BooruPost?> GetPostAsync(long postId, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();
}

public class TagSurveyorTests
{
    [Fact]
    public async Task SurveyAsync_MergesSitesDisagreeingOnCategory_IntoOneRowByRawName()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            try
            {
                // Danbooru files "elvaan" as General (unprefixed); Gelbooru files the same
                // raw tag as Character ("character:elvaan"). Before the fix, this produced
                // two separate eligible survey rows that collided downstream when keyed by
                // raw name (DatasetCrawler/TagRefresher's eligibleTagIdentities lookup).
                var clients = new List<IBooruClient>
                {
                    new FakeSurveyClient("danbooru", [new BooruTagCount("elvaan", 1000)]),
                    new FakeSurveyClient("gelbooru", [new BooruTagCount("character:elvaan", 2000)]),
                };

                var summary = await TagSurveyor.SurveyAsync(db, clients, minImages: 1, maxImages: 100);

                Assert.Equal(1, summary.EligibleTagCount);

                var allTags = await db.GetAllSurveyedTagsAsync();
                var survived = Assert.Single(allTags);
                Assert.Equal("elvaan", TagCategoryNaming.RawName(survived.Name));
                Assert.Equal(1000, survived.DanbooruCount);
                Assert.Equal(2000, survived.GelbooruCount);
                // Danbooru's categorization wins on disagreement — see TagSurveyor.SurveyAsync.
                Assert.Equal("elvaan", survived.Name);
            }
            finally
            {
                db.Dispose();
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task SurveyAsync_UsesGelbooruCategory_WhenOnlyGelbooruSawTheTag()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            try
            {
                var clients = new List<IBooruClient>
                {
                    new FakeSurveyClient("danbooru", []),
                    new FakeSurveyClient("gelbooru", [new BooruTagCount("character:frieren", 500)]),
                };

                await TagSurveyor.SurveyAsync(db, clients, minImages: 1, maxImages: 100);

                var allTags = await db.GetAllSurveyedTagsAsync();
                var survived = Assert.Single(allTags);
                Assert.Equal("character:frieren", survived.Name);
                Assert.Null(survived.DanbooruCount);
                Assert.Equal(500, survived.GelbooruCount);
            }
            finally
            {
                db.Dispose();
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task SurveyAsync_FoldsAliasedGelbooruSpelling_IntoDanbooruCanonicalTag()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            try
            {
                // Danbooru's real, current tag is "headpat"; Gelbooru still reports the
                // deprecated "head_pat" spelling as its own live tag. Without alias
                // knowledge these would survey as two unrelated eligible tags — see
                // DanbooruClient.ListActiveTagAliasesAsync's doc comment for why that then
                // breaks quota tracking during a live crawl.
                var clients = new List<IBooruClient>
                {
                    new FakeSurveyClient("danbooru", [new BooruTagCount("headpat", 23846)]),
                    new FakeSurveyClient("gelbooru", [new BooruTagCount("head_pat", 1020)]),
                };
                var tagAliases = new Dictionary<string, string> { ["head_pat"] = "headpat" };

                var summary = await TagSurveyor.SurveyAsync(db, clients, minImages: 1, maxImages: 100, tagAliases: tagAliases);

                Assert.Equal(1, summary.EligibleTagCount);
                var allTags = await db.GetAllSurveyedTagsAsync();
                var survived = Assert.Single(allTags);
                Assert.Equal("headpat", survived.Name);
                Assert.Equal(23846, survived.DanbooruCount);
                Assert.Equal(1020, survived.GelbooruCount);
            }
            finally
            {
                db.Dispose();
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task SurveyAsync_DeletesStaleAliasAntecedentRow_FromABeforeTheAliasWasKnownSurvey()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var db = await CrawlDatabase.OpenOrCreateAsync(directory);
            try
            {
                // Simulates a crawl.sqlite surveyed before tag-alias resolution existed:
                // "head_pat" got its own row, eligible, with real per-site counts —
                // exactly the leftover this survey must clean up once it learns head_pat
                // is really just headpat, so a re-survey doesn't leave the crawler still
                // iterating a tag it can now never earn credit toward (see
                // TagSurveyor.SurveyAsync's tagAliases doc comment).
                await db.UpsertTagSurveysAsync(
                    [("head_pat", null, 1020, true)], DateTimeOffset.UtcNow, onRowWritten: null);

                var clients = new List<IBooruClient>
                {
                    new FakeSurveyClient("danbooru", [new BooruTagCount("headpat", 23846)]),
                    new FakeSurveyClient("gelbooru", [new BooruTagCount("headpat", 20000)]),
                };
                var tagAliases = new Dictionary<string, string> { ["head_pat"] = "headpat" };

                await TagSurveyor.SurveyAsync(db, clients, minImages: 1, maxImages: 100, tagAliases: tagAliases);

                var allTags = await db.GetAllSurveyedTagsAsync();
                Assert.DoesNotContain(allTags, t => t.Name == "head_pat");
                var survived = Assert.Single(allTags);
                Assert.Equal("headpat", survived.Name);
            }
            finally
            {
                db.Dispose();
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
