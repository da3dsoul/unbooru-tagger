using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

/// <summary>
/// Covers <see cref="TagRowMutations.EligibleIdentities"/>'s two genuinely different input
/// shapes and the incident that proved they weren't both handled:
///
/// <list type="bullet">
/// <item>Live-crawl callers (<c>DatasetCrawler</c>, <c>TagRefresher</c>) pass
/// <c>BooruPost.Tags</c> — genuinely raw, never-prefixed booru names straight off a site's
/// own API.</item>
/// <item><c>VocabularyRebuilder</c> replays <c>crawl.sqlite</c>'s <c>ImageSources.Tags</c>
/// snapshots, which are the OUTPUT of an earlier <c>EligibleIdentities</c> call and are
/// therefore already category-prefixed identities (<c>character:megurine_luka</c>).</item>
/// </list>
///
/// The lookup dictionary is keyed by raw name, so before the fix the second shape missed on
/// every non-General tag and was silently dropped by the <c>OfType&lt;string&gt;()</c>
/// filter — no throw, no log. A real <c>rebuild-vocabulary --apply</c> run took
/// <c>series:vocaloid</c> from 12,634 images to 0 and zeroed 12,642 of 24,894 tags this way.
/// </summary>
public class TagRowMutationsTests
{
    private static TagSurveyResult Tag(string identity, int count) => new(identity, count, count);

    /// <summary>The survey shape used by most tests below: a General tag, one of each prefixed category, and one prefixed tag sitting below any realistic quota.</summary>
    private static readonly TagSurveyResult[] Survey =
    [
        Tag("1girl", 5_000_000),
        Tag("character:megurine_luka", 483),
        Tag("series:vocaloid", 12_634),
        Tag("artist:kamiya_maneki", 210),
        Tag("meta:romaji_commentary", 90_000), // eligible by count; killed by the meta blanket
        Tag("character:barely_seen_oc", 3),    // eligible by category; killed by the quota
    ];

    private static Dictionary<string, string> BuildLookup(
        IEnumerable<TagSurveyResult>? survey = null,
        int minImages = 100,
        TagExclusionRules? exclusions = null,
        IReadOnlyDictionary<string, string>? aliases = null) =>
        TagRowMutations.BuildEligibleIdentities(
            (survey ?? Survey).Where(t => TagEligibility.IsEligible(t, minImages, exclusions)),
            aliases);

    private static TagExclusionRules MetaBlanketOnly() => new(new HashSet<string>(StringComparer.Ordinal), new HashSet<string>(StringComparer.Ordinal));

    private static List<string> Resolve(IEnumerable<string> tags, IReadOnlyDictionary<string, string> lookup) =>
        TagRowMutations.EligibleIdentities(tags, lookup).ToList();

    /// <summary>
    /// The dictionary's keys are what make stripping-before-lookup provably safe: they are
    /// always raw names, which by this project's hard invariant can never contain <c>:</c>.
    /// A prefixed input therefore can only ever MISS a verbatim lookup — which is why
    /// "strip, then look up" is exactly equivalent to "look up verbatim, then strip and
    /// retry", and why the fix can't change behaviour for raw input.
    /// </summary>
    [Fact]
    public void BuildEligibleIdentities_OnlyEverProducesRawUnprefixedKeys()
    {
        var lookup = BuildLookup(exclusions: MetaBlanketOnly());

        Assert.NotEmpty(lookup);
        Assert.All(lookup.Keys, key => Assert.DoesNotContain(':', key));
        Assert.Equal("character:megurine_luka", lookup["megurine_luka"]);
        Assert.Equal("series:vocaloid", lookup["vocaloid"]);
        Assert.Equal("1girl", lookup["1girl"]);
    }

    /// <summary>
    /// The exact regression the fix targets: a <c>VocabularyRebuilder</c> replay feeds
    /// already-resolved identities back in, and every one that's still eligible under that
    /// same identity must resolve straight back to itself.
    /// </summary>
    [Fact]
    public void EligibleIdentities_ResolvesAnAlreadyPrefixedIdentityBackToItself()
    {
        var lookup = BuildLookup(exclusions: MetaBlanketOnly());

        var resolved = Resolve(["1girl", "character:megurine_luka", "series:vocaloid", "artist:kamiya_maneki"], lookup);

        Assert.Equal(["1girl", "character:megurine_luka", "series:vocaloid", "artist:kamiya_maneki"], resolved);
    }

    /// <summary>The live-crawl shape: genuinely raw API names, which must keep resolving exactly as they always did.</summary>
    [Fact]
    public void EligibleIdentities_StillResolvesGenuinelyRawApiTagsUnchanged()
    {
        var lookup = BuildLookup(exclusions: MetaBlanketOnly());

        var resolved = Resolve(["1girl", "megurine_luka", "vocaloid", "kamiya_maneki"], lookup);

        Assert.Equal(["1girl", "character:megurine_luka", "series:vocaloid", "artist:kamiya_maneki"], resolved);
    }

    /// <summary>
    /// Raw and prefixed forms of the same tag are interchangeable inputs — the only
    /// difference between a live post and a stored snapshot — and must land on the same
    /// identity. Belt-and-braces against a future "fix" that special-cases one caller.
    /// </summary>
    [Fact]
    public void EligibleIdentities_TreatsRawAndPrefixedFormsOfTheSameTagIdentically()
    {
        var lookup = BuildLookup(exclusions: MetaBlanketOnly());

        Assert.Equal(
            Resolve(["megurine_luka", "vocaloid", "kamiya_maneki", "1girl"], lookup),
            Resolve(["character:megurine_luka", "series:vocaloid", "artist:kamiya_maneki", "1girl"], lookup));
    }

    /// <summary>
    /// The one case that accidentally worked before the fix (a raw General-shaped string
    /// hits the raw-keyed dictionary directly) — a tag stored under the wrong category and
    /// since reconciled by the survey must be re-credited to the identity in force today.
    /// </summary>
    [Fact]
    public void EligibleIdentities_ReResolvesARawTagWhoseSurveyedCategoryHasSinceBeenCorrected()
    {
        // Stored back when the survey filed "inkling" as General; the survey now agrees it's a character.
        var lookup = BuildLookup([Tag("character:inkling", 4_000)]);

        Assert.Equal(["character:inkling"], Resolve(["inkling"], lookup));
    }

    /// <summary>The mirror image of the above, only reachable at all once prefixed input resolves: a tag stored as a Character that the survey has since reconciled to General.</summary>
    [Fact]
    public void EligibleIdentities_ReResolvesAPrefixedIdentityWhoseSurveyedCategoryHasSinceBeenCorrected()
    {
        var lookup = BuildLookup([Tag("elvaan", 4_000)]);

        Assert.Equal(["elvaan"], Resolve(["character:elvaan"], lookup));
    }

    /// <summary>
    /// Stripping must never become "default to the raw name": a tag the current survey no
    /// longer considers eligible has to be dropped outright, whichever shape it arrives in.
    /// <c>meta:romaji_commentary</c> is excluded by the meta blanket even though its post
    /// count is enormous; <c>character:barely_seen_oc</c> falls under the quota.
    /// </summary>
    [Fact]
    public void EligibleIdentities_DropsTagsTheCurrentSurveyNoLongerConsidersEligible()
    {
        var lookup = BuildLookup(exclusions: MetaBlanketOnly());

        Assert.Empty(Resolve(
            [
                "meta:romaji_commentary", "romaji_commentary",       // excluded by the meta blanket
                "character:barely_seen_oc", "barely_seen_oc",        // below --min-images
                "character:never_surveyed_at_all", "never_surveyed", // no survey row at all
            ],
            lookup));
    }

    /// <summary>A meta tag carved back out of the blanket is genuinely eligible again, and must resolve from its stored prefixed identity too — proving the drop above is about eligibility, not about the <c>meta:</c> prefix.</summary>
    [Fact]
    public void EligibleIdentities_ResolvesAMetaTagThatAnExclusionCarveOutPutsBackIn()
    {
        var lookup = BuildLookup(
            [Tag("meta:traditional_media", 200_000)],
            exclusions: new TagExclusionRules(
                new HashSet<string>(StringComparer.Ordinal),
                new HashSet<string>(["meta:traditional_media"], StringComparer.Ordinal)));

        Assert.Equal(["meta:traditional_media"], Resolve(["meta:traditional_media"], lookup));
        Assert.Equal(["meta:traditional_media"], Resolve(["traditional_media"], lookup));
    }

    /// <summary>The alias-antecedent backfill still works for both shapes — an antecedent is itself a raw name, so a stored General identity spelled the old way resolves to the merged tag's current identity.</summary>
    [Fact]
    public void EligibleIdentities_StillFollowsAliasAntecedentsAfterStripping()
    {
        var aliases = new Dictionary<string, string>(StringComparer.Ordinal) { ["head_pat"] = "headpat", ["luka"] = "megurine_luka" };
        var lookup = BuildLookup([Tag("headpat", 50_000), Tag("character:megurine_luka", 483)], aliases: aliases);

        Assert.Equal(["headpat"], Resolve(["head_pat"], lookup));
        Assert.Equal(["character:megurine_luka"], Resolve(["luka"], lookup));
        // ...and the consequent's own already-resolved identity keeps resolving too.
        Assert.Equal(["character:megurine_luka"], Resolve(["character:megurine_luka"], lookup));
    }

    /// <summary>
    /// <c>BuildEligibleIdentities</c> already collapses a raw name that two sites filed
    /// under different categories down to a single winning identity, so stripping can never
    /// introduce an ambiguity that wasn't already resolved there — a stored identity for
    /// EITHER of the two stale rows lands on the same, single winner.
    /// </summary>
    [Fact]
    public void EligibleIdentities_CollapsesCrossSiteCategoryDisagreementToTheHigherCountedIdentity()
    {
        var lookup = BuildLookup([Tag("elvaan", 120), Tag("character:elvaan", 9_000)]);

        Assert.Equal(["character:elvaan"], Resolve(["elvaan"], lookup));
        Assert.Equal(["character:elvaan"], Resolve(["character:elvaan"], lookup));
    }

    // ---------------------------------------------------------------------------------
    // VocabularyRebuilder end-to-end — the regression test that would have caught the
    // incident, exercised through the real crawl.sqlite schema rather than the pure helper.
    // ---------------------------------------------------------------------------------

    private static PendingNewImage NewImage(string md5, int cacheRowIndex, long postId, IReadOnlyCollection<string> storedTags) =>
        new(md5, cacheRowIndex, 100, 100, DateTimeOffset.UtcNow, "danbooru", postId,
            $"https://example/{postId}.jpg", "g", DateTimeOffset.UtcNow, storedTags, PHash: 0);

    /// <summary>
    /// The end-to-end regression test for the incident. Every image's stored
    /// <c>ImageSources.Tags</c> snapshot is in the shape <c>DatasetCrawler</c> actually
    /// writes — already-resolved, category-prefixed identities — and rebuilding from it
    /// must reproduce each tag's real image count instead of zeroing every prefixed tag.
    /// </summary>
    [Fact]
    public async Task VocabularyRebuilder_CreditsAlreadyPrefixedIdentitiesFromStoredSnapshots()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"rebuild-prefixed-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        var db = await CrawlDatabase.OpenOrCreateAsync(dir);
        try
        {
            await db.UpsertTagSurveysAsync(
                Survey.Concat([Tag("character:inkling", 4_000)])
                    .Select(t => (t.Name, t.DanbooruCount, t.GelbooruCount, Eligible: true, Deprecated: false)),
                DateTimeOffset.UtcNow, onRowWritten: null);

            // Exactly what DatasetCrawler stores: identities, not raw names. The last two
            // entries on image 0 are the ones a rebuild must correctly DROP (excluded /
            // under quota), and "inkling" is a stale raw General spelling the survey has
            // since reconciled to character:inkling.
            await db.CommitPendingImagesAsync(
                [
                    NewImage("md5-a", 0, 1, ["1girl", "character:megurine_luka", "series:vocaloid", "meta:romaji_commentary", "character:barely_seen_oc"]),
                    NewImage("md5-b", 1, 2, ["1girl", "series:vocaloid", "artist:kamiya_maneki"]),
                    NewImage("md5-c", 2, 3, ["1girl", "inkling"]),
                ],
                [], [], CancellationToken.None);

            // A seed snapshot carrying deliberately stale counts, to prove they're recomputed.
            var seed = TagVocabulary.CreateEmpty();
            foreach (var tag in new[] { "1girl", "character:megurine_luka", "series:vocaloid" })
                seed.AddTag(tag).ImageCount = 999;
            var pinnedVocaloidRow = seed.Records.Single(r => r.Tag == "series:vocaloid").RowIndex;

            var (rows, result) = await VocabularyRebuilder.Rebuild(
                db, seed, minImages: 100, excludedTags: MetaBlanketOnly(), tagAliases: null, confirmedImageCount: 3);

            int Count(string tag)
            {
                Assert.True(seed.TryGet(tag, out var record), $"'{tag}' is missing from the rebuilt vocabulary entirely.");
                return record.ImageCount;
            }

            // The heart of it: prefixed identities survive the replay with real counts.
            Assert.Equal(2, Count("series:vocaloid"));
            Assert.Equal(1, Count("character:megurine_luka"));
            Assert.Equal(1, Count("artist:kamiya_maneki"));
            Assert.Equal(3, Count("1girl"));
            // The stale raw spelling is re-credited to the identity the survey uses today.
            Assert.Equal(1, Count("character:inkling"));

            // ...and genuinely-ineligible tags are dropped, not defaulted to anything.
            Assert.False(seed.TryGet("meta:romaji_commentary", out _));
            Assert.False(seed.TryGet("romaji_commentary", out _));
            Assert.False(seed.TryGet("character:barely_seen_oc", out _));
            Assert.False(seed.TryGet("inkling", out _));

            // Seeded rows keep their exact indices (trained embeddings depend on it).
            Assert.Equal(pinnedVocaloidRow, seed.Records.Single(r => r.Tag == "series:vocaloid").RowIndex);

            Assert.Equal(3, result.ImagesWithSourceData);
            Assert.Equal(0, result.ImagesWithoutSourceData);
            Assert.Equal(3, result.SeedTagCount);
            Assert.Equal(5, result.FinalTagCount); // + artist:kamiya_maneki, character:inkling

            // The per-image tag rows themselves, decoded back to tag names.
            var namesByCacheRow = rows.ToDictionary(
                r => r.CacheRowIndex,
                r => r.TagRows.Select(i => seed.GetByRowIndex(i).Tag).OrderBy(t => t, StringComparer.Ordinal).ToList());

            Assert.Equal(["1girl", "character:megurine_luka", "series:vocaloid"], namesByCacheRow[0]);
            Assert.Equal(["1girl", "artist:kamiya_maneki", "series:vocaloid"], namesByCacheRow[1]);
            Assert.Equal(["1girl", "character:inkling"], namesByCacheRow[2]);
        }
        finally
        {
            db.Dispose();
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// The alias backfill has to survive the replay too: an image whose stored snapshot
    /// still spells a tag the pre-merge way must be credited to the merged identity, not
    /// dropped — the same guarantee <c>refresh-tags</c> relies on, but through the rebuild.
    /// </summary>
    [Fact]
    public async Task VocabularyRebuilder_CreditsStoredSnapshotsThroughTagAliases()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"rebuild-aliases-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        var db = await CrawlDatabase.OpenOrCreateAsync(dir);
        try
        {
            await db.UpsertTagSurveysAsync(
                new[] { Tag("headpat", 50_000), Tag("character:megurine_luka", 483) }
                    .Select(t => (t.Name, t.DanbooruCount, t.GelbooruCount, Eligible: true, Deprecated: false)),
                DateTimeOffset.UtcNow, onRowWritten: null);

            await db.CommitPendingImagesAsync(
                [NewImage("md5-a", 0, 1, ["head_pat", "character:megurine_luka"])],
                [], [], CancellationToken.None);

            var seed = TagVocabulary.CreateEmpty();
            var aliases = new Dictionary<string, string>(StringComparer.Ordinal) { ["head_pat"] = "headpat" };

            await VocabularyRebuilder.Rebuild(db, seed, minImages: 100, excludedTags: MetaBlanketOnly(), aliases, confirmedImageCount: 1);

            Assert.True(seed.TryGet("headpat", out var headpat));
            Assert.Equal(1, headpat.ImageCount);
            Assert.True(seed.TryGet("character:megurine_luka", out var luka));
            Assert.Equal(1, luka.ImageCount);
            Assert.False(seed.TryGet("head_pat", out _));
        }
        finally
        {
            db.Dispose();
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>
    /// An image whose <c>ImageSources.Tags</c> was never captured (a null snapshot) still
    /// has to come back with an empty tag row rather than borrowing anything — unchanged by
    /// this fix, asserted here so the rebuild's two "no tags" paths stay distinguishable.
    /// </summary>
    [Fact]
    public async Task VocabularyRebuilder_LeavesImagesWithoutSourceDataEmpty()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"rebuild-nosource-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        var db = await CrawlDatabase.OpenOrCreateAsync(dir);
        try
        {
            await db.UpsertTagSurveysAsync(
                new[] { Tag("series:vocaloid", 12_634) }.Select(t => (t.Name, t.DanbooruCount, t.GelbooruCount, Eligible: true, Deprecated: false)),
                DateTimeOffset.UtcNow, onRowWritten: null);

            await db.CommitPendingImagesAsync([NewImage("md5-a", 0, 1, ["series:vocaloid"])], [], [], CancellationToken.None);

            var seed = TagVocabulary.CreateEmpty();
            // confirmedImageCount is 2: cache row 1 exists in the pixel cache but crawl.sqlite knows nothing about it.
            var (rows, result) = await VocabularyRebuilder.Rebuild(db, seed, minImages: 100, excludedTags: MetaBlanketOnly(), tagAliases: null, confirmedImageCount: 2);

            Assert.Equal(1, result.ImagesWithSourceData);
            Assert.Equal(1, result.ImagesWithoutSourceData);
            Assert.Single(rows.Single(r => r.CacheRowIndex == 0).TagRows);
            Assert.Empty(rows.Single(r => r.CacheRowIndex == 1).TagRows);
            Assert.Equal(1, seed.Records.Single(r => r.Tag == "series:vocaloid").ImageCount);
        }
        finally
        {
            db.Dispose();
            Directory.Delete(dir, recursive: true);
        }
    }
}
