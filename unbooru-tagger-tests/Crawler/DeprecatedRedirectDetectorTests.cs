using System.Globalization;
using System.Text.Json;
using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

/// <summary>
/// Answers <see cref="IBooruClient.ListPostsAsync"/> out of an in-memory tag -&gt; posts map,
/// reproducing the id-descending cursor pagination both real clients implement (a cursor is
/// the last post id of the previous page; the next page is everything strictly older than
/// it), including their shared "a full page means there might be more, a short page means
/// there isn't" rule for <see cref="BooruPostPage.NextCursor"/> — the exact behaviour
/// <see cref="DeprecatedRedirectDetector"/>'s exhaustion comparison depends on, so faking it
/// loosely would let a broken comparison pass. Records every query so a test can prove which
/// searches were (and crucially, were NOT) spent. Everything the detector doesn't call is
/// unsupported on purpose.
/// </summary>
internal sealed class FakeSearchClient(string siteName, Dictionary<string, IReadOnlyList<BooruPost>> postsByTag, int pageSize = 3) : IBooruClient
{
    public string SiteName => siteName;
    public int PageSize => pageSize;
    public IRateLimiter RateLimiter { get; } = new ImmediateRateLimiter();

    /// <summary>Every tag query this client was asked for, in order — including repeats across pages.</summary>
    public List<string> Queries { get; } = [];

    public IAsyncEnumerable<BooruTagCount> ListTagsByCountDescendingAsync(CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<BooruPost?> GetPostAsync(long postId, CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public Task<BooruPostPage> ListPostsAsync(string tagQuery, string? cursor, CancellationToken cancellationToken = default)
    {
        Queries.Add(tagQuery);

        var ordered = (postsByTag.GetValueOrDefault(tagQuery) ?? [])
            .OrderByDescending(post => post.PostId)
            .ToList();
        var remaining = cursor is null
            ? ordered
            : ordered.Where(post => post.PostId < long.Parse(cursor, CultureInfo.InvariantCulture)).ToList();

        var page = remaining.Take(pageSize).ToList();
        var nextCursor = page.Count == pageSize ? page[^1].PostId.ToString(CultureInfo.InvariantCulture) : null;
        return Task.FromResult(new BooruPostPage(page, nextCursor));
    }
}

public class DeprecatedRedirectDetectorTests
{
    /// <summary>Small enough that a handful of fake posts is a "big enough" sample, while still exercising multi-page sampling (3 posts/page, 2 pages).</summary>
    private static readonly DeprecatedRedirectOptions TestOptions = new() { SamplePages = 2, MinSamplePosts = 4, MaxCandidates = 8 };

    private static BooruPost MakePost(long id, params string[] tags) =>
        new(id, $"md5-{id}", new Uri("https://example/x.jpg"), tags, "g", DateTimeOffset.UnixEpoch, 100, 100);

    /// <summary>Six posts (ids 106..101, id-descending) all carrying <paramref name="tags"/>.</summary>
    private static IReadOnlyList<BooruPost> Posts(params string[] tags) =>
        Enumerable.Range(0, 6).Select(i => MakePost(106 - i, tags)).ToList();

    private static TagSurveyResult DeprecatedOnGelbooru(string name) => new(name, null, 1000, Deprecated: true);

    [Fact]
    public async Task DetectAsync_ConfirmsARedirect_WhenACandidateReproducesTheExactPostSequence()
    {
        // The real, live-confirmed case this command exists for: Gelbooru's deprecated
        // 'pussy_licking' returns byte-identical search results to 'cunnilingus', while
        // appearing in no alias table anywhere. Note the returned posts carry 'cunnilingus'
        // and NOT 'pussy_licking' — that's what a redirected search looks like.
        var shared = Posts("cunnilingus", "1girl");
        var client = new FakeSearchClient("gelbooru", new Dictionary<string, IReadOnlyList<BooruPost>>
        {
            ["pussy_licking"] = shared,
            ["cunnilingus"] = shared,
            ["1girl"] = shared.Concat([MakePost(200, "1girl")]).ToList(), // a genuinely broader tag: co-occurs everywhere, but its own search differs
        });

        var result = await DeprecatedRedirectDetector.DetectAsync(
            [DeprecatedOnGelbooru("pussy_licking")],
            new Dictionary<string, IBooruClient> { ["gelbooru"] = client },
            knownAliases: null,
            TestOptions);

        Assert.Equal("cunnilingus", Assert.Single(result.NewAliases).Value);
        Assert.Equal("pussy_licking", Assert.Single(result.NewAliases).Key);

        var finding = Assert.Single(result.Findings);
        Assert.Equal(DeprecatedRedirectOutcome.Confirmed, finding.Outcome);
        Assert.Equal("cunnilingus", finding.Target);
        Assert.Equal("gelbooru", finding.SiteName);
        Assert.Equal(6, finding.SampledPosts);
        // Both co-occurring tags were candidates; only one survived verification.
        Assert.Equal(new[] { "1girl", "cunnilingus" }, finding.Candidates);
        Assert.Equal(new[] { "cunnilingus" }, finding.VerifiedTargets);
    }

    [Fact]
    public async Task DetectAsync_SkipsATagWithNoResults_WithoutWritingAnAlias()
    {
        var client = new FakeSearchClient("gelbooru", new Dictionary<string, IReadOnlyList<BooruPost>>());

        var result = await DeprecatedRedirectDetector.DetectAsync(
            [DeprecatedOnGelbooru("dead_tag")],
            new Dictionary<string, IBooruClient> { ["gelbooru"] = client },
            knownAliases: null,
            TestOptions);

        Assert.Empty(result.NewAliases);
        Assert.Equal(DeprecatedRedirectOutcome.NoResults, Assert.Single(result.Findings).Outcome);
        // One request to discover there's nothing there, and nothing spent verifying.
        Assert.Equal(new[] { "dead_tag" }, client.Queries);
    }

    [Fact]
    public async Task DetectAsync_RejectsACooccurringCandidate_WhoseOwnResultsDoNotMatchExactly()
    {
        // 'cunnilingus' is on 100% of the deprecated tag's results, which is exactly what a
        // redirect target looks like — but its own search returns a post the deprecated
        // tag's search doesn't, so the two are provably different queries. Co-occurrence
        // alone must never be enough.
        var deprecatedResults = Posts("cunnilingus");
        var client = new FakeSearchClient("gelbooru", new Dictionary<string, IReadOnlyList<BooruPost>>
        {
            ["oral_sex"] = deprecatedResults,
            ["cunnilingus"] = deprecatedResults.Concat([MakePost(999, "cunnilingus")]).ToList(),
        });

        var result = await DeprecatedRedirectDetector.DetectAsync(
            [DeprecatedOnGelbooru("oral_sex")],
            new Dictionary<string, IBooruClient> { ["gelbooru"] = client },
            knownAliases: null,
            TestOptions);

        Assert.Empty(result.NewAliases);
        var finding = Assert.Single(result.Findings);
        Assert.Equal(DeprecatedRedirectOutcome.NoCandidateVerified, finding.Outcome);
        Assert.Null(finding.Target);
        Assert.Equal(new[] { "cunnilingus" }, finding.Candidates);
        Assert.Empty(finding.VerifiedTargets);
        // Rejected on the candidate's very first page — verification never pays for a full
        // sample it already knows can't match.
        Assert.Equal(new[] { "oral_sex", "oral_sex", "cunnilingus" }, client.Queries);
    }

    [Fact]
    public async Task DetectAsync_PicksTheOneCandidateThatVerifies_WhenSeveralCooccurOnEveryPost()
    {
        var shared = Posts("cunnilingus", "yuri", "nude");
        var client = new FakeSearchClient("gelbooru", new Dictionary<string, IReadOnlyList<BooruPost>>
        {
            ["pussy_licking"] = shared,
            ["cunnilingus"] = shared,
            ["yuri"] = shared.Concat([MakePost(300, "yuri")]).ToList(),
            ["nude"] = shared.Take(4).ToList(),
        });

        var result = await DeprecatedRedirectDetector.DetectAsync(
            [DeprecatedOnGelbooru("pussy_licking")],
            new Dictionary<string, IBooruClient> { ["gelbooru"] = client },
            knownAliases: null,
            TestOptions);

        Assert.Equal("cunnilingus", Assert.Single(result.NewAliases).Value);
        var finding = Assert.Single(result.Findings);
        Assert.Equal(DeprecatedRedirectOutcome.Confirmed, finding.Outcome);
        Assert.Equal(new[] { "cunnilingus", "nude", "yuri" }, finding.Candidates);
        Assert.Equal(new[] { "cunnilingus" }, finding.VerifiedTargets);
    }

    [Fact]
    public async Task DetectAsync_SkipsAKnownAliasAntecedent_WithoutSpendingASingleRequest()
    {
        // Already officially aliased by the site, so prune-aliased-tags/TagSurveyor already
        // handle it correctly — re-deriving it empirically would be two wasted live searches
        // per tag across a corpus with tens of thousands of official aliases in it.
        var client = new FakeSearchClient("gelbooru", new Dictionary<string, IReadOnlyList<BooruPost>>
        {
            ["curvy_figure"] = Posts("curvy"),
            ["curvy"] = Posts("curvy"),
        });

        var result = await DeprecatedRedirectDetector.DetectAsync(
            [DeprecatedOnGelbooru("curvy_figure")],
            new Dictionary<string, IBooruClient> { ["gelbooru"] = client },
            knownAliases: new Dictionary<string, string> { ["curvy_figure"] = "curvy" },
            TestOptions);

        Assert.Empty(result.NewAliases);
        var finding = Assert.Single(result.Findings);
        Assert.Equal(DeprecatedRedirectOutcome.AlreadyAliased, finding.Outcome);
        Assert.Null(finding.SiteName);
        Assert.Empty(client.Queries);
        Assert.Equal(0, result.RequestsMade);
    }

    [Fact]
    public async Task DetectAsync_LeavesATagUnresolved_WhenEverySampledPostLiterallyCarriesIt()
    {
        // The single most important false-positive guard: a genuine (un-redirected) tag
        // search returns only posts that really carry the tag. 'blue_eyes' here is a real
        // narrower tag that happens to always co-occur with 'blue_hair', and their samples
        // would match exactly — but the results carrying 'blue_eyes' itself proves the site
        // ran a real search for it rather than redirecting to something else.
        var shared = Posts("blue_eyes", "blue_hair");
        var client = new FakeSearchClient("gelbooru", new Dictionary<string, IReadOnlyList<BooruPost>>
        {
            ["blue_eyes"] = shared,
            ["blue_hair"] = shared,
        });

        var result = await DeprecatedRedirectDetector.DetectAsync(
            [DeprecatedOnGelbooru("blue_eyes")],
            new Dictionary<string, IBooruClient> { ["gelbooru"] = client },
            knownAliases: null,
            TestOptions);

        Assert.Empty(result.NewAliases);
        Assert.Equal(DeprecatedRedirectOutcome.SearchLooksGenuine, Assert.Single(result.Findings).Outcome);
        // Decided from the sample alone — no candidate was ever verified.
        Assert.Equal(new[] { "blue_eyes", "blue_eyes" }, client.Queries);
    }

    [Fact]
    public async Task DetectAsync_LeavesATagUnresolved_WhenSeveralCandidatesAllMatchExactly()
    {
        var shared = Posts("cunnilingus", "oral");
        var client = new FakeSearchClient("gelbooru", new Dictionary<string, IReadOnlyList<BooruPost>>
        {
            ["pussy_licking"] = shared,
            ["cunnilingus"] = shared,
            ["oral"] = shared,
        });

        var result = await DeprecatedRedirectDetector.DetectAsync(
            [DeprecatedOnGelbooru("pussy_licking")],
            new Dictionary<string, IBooruClient> { ["gelbooru"] = client },
            knownAliases: null,
            TestOptions);

        Assert.Empty(result.NewAliases);
        var finding = Assert.Single(result.Findings);
        Assert.Equal(DeprecatedRedirectOutcome.AmbiguousCandidates, finding.Outcome);
        Assert.Equal(new[] { "cunnilingus", "oral" }, finding.VerifiedTargets);
        Assert.Null(finding.Target);
    }

    [Fact]
    public async Task DetectAsync_RefusesToWriteAnAlias_ThatWouldCloseACycle()
    {
        var shared = Posts("cunnilingus");
        var client = new FakeSearchClient("gelbooru", new Dictionary<string, IReadOnlyList<BooruPost>>
        {
            ["pussy_licking"] = shared,
            ["cunnilingus"] = shared,
        });

        var result = await DeprecatedRedirectDetector.DetectAsync(
            [DeprecatedOnGelbooru("pussy_licking")],
            new Dictionary<string, IBooruClient> { ["gelbooru"] = client },
            // The site already publishes the alias in the other direction, so writing
            // pussy_licking -> cunnilingus would make the chain loop forever.
            knownAliases: new Dictionary<string, string> { ["cunnilingus"] = "pussy_licking" },
            TestOptions);

        Assert.Empty(result.NewAliases);
        var finding = Assert.Single(result.Findings);
        Assert.Equal(DeprecatedRedirectOutcome.WouldCreateCycle, finding.Outcome);
        Assert.Equal("cunnilingus", finding.Target); // reported so a human can look, never written
    }

    [Fact]
    public async Task DetectAsync_LeavesATagUnresolved_WhenTheSampleIsTooSmallToProveAnything()
    {
        // Two tags sitting on the same two images match exactly by pure coincidence — which
        // is precisely why a match over a tiny sample is worth nothing.
        var shared = new List<BooruPost> { MakePost(2, "y"), MakePost(1, "y") };
        var client = new FakeSearchClient("gelbooru", new Dictionary<string, IReadOnlyList<BooruPost>>
        {
            ["x"] = shared,
            ["y"] = shared,
        });

        var result = await DeprecatedRedirectDetector.DetectAsync(
            [DeprecatedOnGelbooru("x")],
            new Dictionary<string, IBooruClient> { ["gelbooru"] = client },
            knownAliases: null,
            TestOptions);

        Assert.Empty(result.NewAliases);
        var finding = Assert.Single(result.Findings);
        Assert.Equal(DeprecatedRedirectOutcome.SampleTooSmall, finding.Outcome);
        Assert.Equal(2, finding.SampledPosts);
    }

    [Fact]
    public async Task DetectAsync_ProbesOnlyTheSitesThatActuallySurveyedTheTag()
    {
        var shared = Posts("cunnilingus");
        var gelbooru = new FakeSearchClient("gelbooru", new Dictionary<string, IReadOnlyList<BooruPost>>
        {
            ["pussy_licking"] = shared,
            ["cunnilingus"] = shared,
        });
        var danbooru = new FakeSearchClient("danbooru", new Dictionary<string, IReadOnlyList<BooruPost>>());

        var result = await DeprecatedRedirectDetector.DetectAsync(
            [DeprecatedOnGelbooru("pussy_licking")], // GelbooruCount only — Danbooru never saw this tag
            new Dictionary<string, IBooruClient> { ["danbooru"] = danbooru, ["gelbooru"] = gelbooru },
            knownAliases: null,
            TestOptions);

        Assert.Equal("cunnilingus", Assert.Single(result.NewAliases).Value);
        Assert.Empty(danbooru.Queries);
        Assert.NotEmpty(gelbooru.Queries);
    }

    [Fact]
    public async Task DetectAsync_StopsProbingFurtherSites_OnceOneConfirms()
    {
        // The alias table is one flat map keyed by antecedent, so a second site's answer for
        // the same tag couldn't be stored anyway — probing for it would be pure wasted
        // network. Danbooru goes first, matching TagAliasCache's own "Danbooru wins" rule.
        var shared = Posts("cunnilingus");
        var byTag = new Dictionary<string, IReadOnlyList<BooruPost>>
        {
            ["pussy_licking"] = shared,
            ["cunnilingus"] = shared,
        };
        var danbooru = new FakeSearchClient("danbooru", byTag);
        var gelbooru = new FakeSearchClient("gelbooru", byTag);

        var result = await DeprecatedRedirectDetector.DetectAsync(
            [new TagSurveyResult("pussy_licking", 900, 1000, Deprecated: true)],
            new Dictionary<string, IBooruClient> { ["gelbooru"] = gelbooru, ["danbooru"] = danbooru },
            knownAliases: null,
            TestOptions);

        Assert.Equal("cunnilingus", Assert.Single(result.NewAliases).Value);
        Assert.Equal("danbooru", Assert.Single(result.Findings).SiteName);
        Assert.NotEmpty(danbooru.Queries);
        Assert.Empty(gelbooru.Queries);
    }

    [Fact]
    public async Task DetectAsync_SkipsATagWithMoreCandidatesThanTheCapAllows_RatherThanVerifyingAnArbitrarySubset()
    {
        var shared = Posts("a", "b", "c");
        var client = new FakeSearchClient("gelbooru", new Dictionary<string, IReadOnlyList<BooruPost>>
        {
            ["x"] = shared,
            ["a"] = shared,
        });

        var result = await DeprecatedRedirectDetector.DetectAsync(
            [DeprecatedOnGelbooru("x")],
            new Dictionary<string, IBooruClient> { ["gelbooru"] = client },
            knownAliases: null,
            TestOptions with { MaxCandidates = 2 });

        Assert.Empty(result.NewAliases);
        var finding = Assert.Single(result.Findings);
        Assert.Equal(DeprecatedRedirectOutcome.TooManyCandidates, finding.Outcome);
        Assert.Equal(3, finding.Candidates.Count);
        // 'a' would have verified, but was never even tried — the sample is admitted to be
        // too weak to discriminate, and reporting that beats guessing.
        Assert.Equal(new[] { "x", "x" }, client.Queries);
    }

    [Fact]
    public async Task DetectAsync_WritesNothingToDisk_WhichIsWhatMakesTheCommandsDryRunADryRun()
    {
        // detect-deprecated-redirects' dry run is exactly "run DetectAsync, print the table,
        // return" — the writes live entirely in TagAliasCache.MergeIntoCacheAsync/
        // SaveDetectedRedirectsAsync, which only --apply reaches. This pins the property that
        // makes that safe: detection itself has no write side effect at all, so no future
        // change can quietly give the dry run one.
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var shared = Posts("cunnilingus");
            var client = new FakeSearchClient("gelbooru", new Dictionary<string, IReadOnlyList<BooruPost>>
            {
                ["pussy_licking"] = shared,
                ["cunnilingus"] = shared,
            });

            var result = await DeprecatedRedirectDetector.DetectAsync(
                [DeprecatedOnGelbooru("pussy_licking")],
                new Dictionary<string, IBooruClient> { ["gelbooru"] = client },
                knownAliases: null,
                TestOptions);

            Assert.NotEmpty(result.NewAliases); // it really did have something it could have written
            Assert.Empty(Directory.EnumerateFileSystemEntries(directory));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>
    /// Builds the shape a search FALLBACK produces: <paramref name="antecedentCount"/>
    /// unrelated deprecated tags whose searches all return the exact same posts as the real
    /// tag <paramref name="target"/>. Every one of them verifies legitimately — the match IS
    /// exact — so only the run-level convergence can tell them apart from real redirects.
    /// </summary>
    private static (List<TagSurveyResult> Tags, FakeSearchClient Client) FallbackHub(string target, int antecedentCount)
    {
        var shared = Posts(target);
        var byTag = new Dictionary<string, IReadOnlyList<BooruPost>>(StringComparer.Ordinal) { [target] = shared };
        var tags = new List<TagSurveyResult>();
        for (var i = 0; i < antecedentCount; i++)
        {
            var name = $"garbage_query_{i}";
            byTag[name] = shared;
            tags.Add(DeprecatedOnGelbooru(name));
        }

        return (tags, new FakeSearchClient("gelbooru", byTag));
    }

    [Fact]
    public async Task DetectAsync_SuppressesEveryConfirmation_WhenFarTooManyDistinctTagsConvergeOnOneTarget()
    {
        // The real false positive this guard exists for: Gelbooru answers any query string it
        // can't parse as a search filter with a shared default result set, byte-identical to a
        // real search for 'bad_tag'. 1,000 of one full run's 3,635 confirmations pointed there.
        // Per-tag verification cannot see it — each exact match is genuinely exact.
        var (tags, client) = FallbackHub("bad_tag", 12);

        var result = await DeprecatedRedirectDetector.DetectAsync(
            tags,
            new Dictionary<string, IBooruClient> { ["gelbooru"] = client },
            knownAliases: null,
            TestOptions with { MaxAntecedentsPerTarget = 5 });

        Assert.Empty(result.NewAliases);
        Assert.Equal(12, result.Findings.Count);
        Assert.All(result.Findings, f => Assert.Equal(DeprecatedRedirectOutcome.SuspectedHubTarget, f.Outcome));
        // The target is kept on the finding, not blanked, so the report can name the hub.
        Assert.All(result.Findings, f => Assert.Equal("bad_tag", f.Target));
    }

    [Fact]
    public async Task DetectAsync_StillWritesATargetThatAbsorbsAModestNumberOfVariants()
    {
        // The legitimate shape the threshold must NOT catch, mirroring the real 'ass'/47 and
        // 'smile'/15 clusters: one real tag genuinely is the redirect target for a pile of
        // typo/synonym antecedents, and every one of those is a correct alias to write.
        var (tags, client) = FallbackHub("ass", 5);

        var result = await DeprecatedRedirectDetector.DetectAsync(
            tags,
            new Dictionary<string, IBooruClient> { ["gelbooru"] = client },
            knownAliases: null,
            TestOptions with { MaxAntecedentsPerTarget = 150 });

        Assert.Equal(5, result.NewAliases.Count);
        Assert.All(result.NewAliases, pair => Assert.Equal("ass", pair.Value));
        Assert.All(result.Findings, f => Assert.Equal(DeprecatedRedirectOutcome.Confirmed, f.Outcome));
    }

    [Theory]
    // Strictly "more than", matching --max-candidates' own off-by-one: a target sitting
    // exactly ON the cap is still trusted; one antecedent past it is not.
    [InlineData(5, true)]
    [InlineData(6, false)]
    public async Task DetectAsync_TrustsATargetExactlyAtTheCap_AndDistrustsTheOneAntecedentPastIt(int antecedentCount, bool expectWritten)
    {
        var (tags, client) = FallbackHub("bad_tag", antecedentCount);

        var result = await DeprecatedRedirectDetector.DetectAsync(
            tags,
            new Dictionary<string, IBooruClient> { ["gelbooru"] = client },
            knownAliases: null,
            TestOptions with { MaxAntecedentsPerTarget = 5 });

        Assert.Equal(expectWritten ? antecedentCount : 0, result.NewAliases.Count);
        Assert.All(result.Findings, f => Assert.Equal(
            expectWritten ? DeprecatedRedirectOutcome.Confirmed : DeprecatedRedirectOutcome.SuspectedHubTarget,
            f.Outcome));
    }

    [Fact]
    public async Task DetectAsync_SuppressesAHub_WithoutSpendingASingleExtraRequest()
    {
        // The whole point of doing this as a post-pass over findings already collected: the
        // convergence itself IS the evidence, so re-asking the network could only re-confirm
        // the same exact match that caused the problem. Pinned by comparing the identical run
        // with the guard effectively off — request counts must match exactly.
        var (suppressedTags, suppressedClient) = FallbackHub("bad_tag", 8);
        var suppressed = await DeprecatedRedirectDetector.DetectAsync(
            suppressedTags,
            new Dictionary<string, IBooruClient> { ["gelbooru"] = suppressedClient },
            knownAliases: null,
            TestOptions with { MaxAntecedentsPerTarget = 3 });

        var (writtenTags, writtenClient) = FallbackHub("bad_tag", 8);
        var written = await DeprecatedRedirectDetector.DetectAsync(
            writtenTags,
            new Dictionary<string, IBooruClient> { ["gelbooru"] = writtenClient },
            knownAliases: null,
            TestOptions with { MaxAntecedentsPerTarget = 1000 });

        Assert.Empty(suppressed.NewAliases);
        Assert.Equal(8, written.NewAliases.Count);

        Assert.Equal(written.RequestsMade, suppressed.RequestsMade);
        Assert.Equal(writtenClient.Queries, suppressedClient.Queries);
        // ...and the per-probe request accounting is untouched too, not just the total.
        Assert.Equal(
            written.Findings.Select(f => f.RequestsMade),
            suppressed.Findings.Select(f => f.RequestsMade));
    }

    [Fact]
    public async Task DetectAsync_LeavesEveryOtherOutcomeAlone_WhenAHubIsSuppressedInTheSameRun()
    {
        // Only Confirmed findings pointing AT the hub are reclassified. A genuine redirect to
        // a different target in the same run keeps its alias, and the non-writing outcomes
        // keep reporting exactly what they reported before.
        var (hubTags, _) = FallbackHub("bad_tag", 6);
        var hubPosts = Posts("bad_tag");
        var byTag = new Dictionary<string, IReadOnlyList<BooruPost>>(StringComparer.Ordinal) { ["bad_tag"] = hubPosts };
        foreach (var tag in hubTags)
            byTag[tag.Name] = hubPosts;

        // A real, unrelated redirect...
        var realPair = Posts("cunnilingus");
        byTag["pussy_licking"] = realPair;
        byTag["cunnilingus"] = realPair;
        // ...and a tag whose own search is genuine, so it never reaches confirmation at all.
        var genuine = Posts("blue_eyes", "blue_hair");
        byTag["blue_eyes"] = genuine;
        byTag["blue_hair"] = genuine;

        var client = new FakeSearchClient("gelbooru", byTag);
        var tags = hubTags.Concat([DeprecatedOnGelbooru("pussy_licking"), DeprecatedOnGelbooru("blue_eyes")]).ToList();

        var result = await DeprecatedRedirectDetector.DetectAsync(
            tags,
            new Dictionary<string, IBooruClient> { ["gelbooru"] = client },
            knownAliases: null,
            TestOptions with { MaxAntecedentsPerTarget = 3 });

        Assert.Equal("cunnilingus", Assert.Single(result.NewAliases).Value);
        Assert.Equal("pussy_licking", Assert.Single(result.NewAliases).Key);

        var byOutcome = result.Findings.GroupBy(f => f.Outcome).ToDictionary(g => g.Key, g => g.Count());
        Assert.Equal(6, byOutcome[DeprecatedRedirectOutcome.SuspectedHubTarget]);
        Assert.Equal(1, byOutcome[DeprecatedRedirectOutcome.Confirmed]);
        Assert.Equal(1, byOutcome[DeprecatedRedirectOutcome.SearchLooksGenuine]);
    }

    [Fact]
    public async Task MergeIntoCacheAsync_KeepsEveryExistingEntry_AndAddsTheNewOnes_InTheFormatEverythingElseReads()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            // The shape TagAliasCache.FetchAndCacheAsync writes: a flat antecedent ->
            // consequent object, nothing else.
            await File.WriteAllTextAsync(
                Path.Combine(directory, TagAliasCache.FileName),
                JsonSerializer.Serialize(new Dictionary<string, string> { ["head_pat"] = "headpat", ["curvy_figure"] = "curvy" }));

            var added = await TagAliasCache.MergeIntoCacheAsync(
                directory, new Dictionary<string, string> { ["pussy_licking"] = "cunnilingus" });

            Assert.Equal("cunnilingus", Assert.Single(added).Value);

            var reloaded = await TagAliasCache.TryLoadAsync(directory);
            Assert.NotNull(reloaded);
            Assert.Equal(3, reloaded!.Count);
            Assert.Equal("headpat", reloaded["head_pat"]);
            Assert.Equal("curvy", reloaded["curvy_figure"]);
            Assert.Equal("cunnilingus", reloaded["pussy_licking"]);

            // Not just "TryLoadAsync can read it back" — the on-disk bytes must still be a
            // plain Dictionary<string, string>, since that's all TagRowMutations/TagSurveyor
            // ever get handed.
            var raw = JsonSerializer.Deserialize<Dictionary<string, string>>(
                await File.ReadAllTextAsync(Path.Combine(directory, TagAliasCache.FileName)));
            Assert.NotNull(raw);
            Assert.Equal("cunnilingus", TagRowMutations.ResolveAlias("pussy_licking", raw!));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task MergeIntoCacheAsync_LeavesAnAlreadyKnownAntecedentAlone_RatherThanRetargetingIt()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            await File.WriteAllTextAsync(
                Path.Combine(directory, TagAliasCache.FileName),
                JsonSerializer.Serialize(new Dictionary<string, string> { ["head_pat"] = "headpat" }));

            var added = await TagAliasCache.MergeIntoCacheAsync(
                directory, new Dictionary<string, string> { ["head_pat"] = "something_else" });

            Assert.Empty(added);
            var reloaded = await TagAliasCache.TryLoadAsync(directory);
            Assert.Equal("headpat", reloaded!["head_pat"]); // the site's own published alias wins
            Assert.Single(reloaded);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task FetchAndCacheAsync_FoldsPreviouslyDetectedRedirectsBackIn_InsteadOfDiscardingThem()
    {
        // survey-tags/refresh-tags overwrite tag_aliases.json outright on every run. Without
        // this, the first one to run after a detection pass would silently throw away hours
        // of rate-limited work that no site's API can ever re-derive.
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            await TagAliasCache.SaveDetectedRedirectsAsync(
                directory, new Dictionary<string, string> { ["pussy_licking"] = "cunnilingus" });

            const string json = """[ { "antecedent_name": "head_pat", "consequent_name": "headpat", "status": "active" } ]""";
            var httpClient = new HttpClient(new SinglePageThenEmptyHttpMessageHandler(json));
            var clients = new Dictionary<string, IBooruClient> { ["danbooru"] = new DanbooruClient(httpClient, new ImmediateRateLimiter()) };

            var fetched = await TagAliasCache.FetchAndCacheAsync(directory, clients);

            Assert.NotNull(fetched);
            Assert.Equal("headpat", fetched!["head_pat"]);
            Assert.Equal("cunnilingus", fetched["pussy_licking"]);

            var reloaded = await TagAliasCache.TryLoadAsync(directory);
            Assert.Equal("cunnilingus", reloaded!["pussy_licking"]);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task FetchAndCacheAsync_LetsAnOfficialAliasOverrideAPreviouslyDetectedRedirect()
    {
        // If a site later publishes a real alias for a tag this command inferred, the
        // published one is authoritative — the inference was only ever a stand-in for it.
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            await TagAliasCache.SaveDetectedRedirectsAsync(
                directory, new Dictionary<string, string> { ["head_pat"] = "guessed_target" });

            const string json = """[ { "antecedent_name": "head_pat", "consequent_name": "headpat", "status": "active" } ]""";
            var httpClient = new HttpClient(new SinglePageThenEmptyHttpMessageHandler(json));
            var clients = new Dictionary<string, IBooruClient> { ["danbooru"] = new DanbooruClient(httpClient, new ImmediateRateLimiter()) };

            var fetched = await TagAliasCache.FetchAndCacheAsync(directory, clients);

            Assert.Equal("headpat", fetched!["head_pat"]);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
