using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class TagFamiliesTests
{
    // 0 shoes, 1 boots, 2 platform_boots, 3 sandals, 4 no_shoes, 5 character:cinderella, 6 footwear_bow, 7 1girl
    private static readonly string[] Tags = ["shoes", "boots", "platform_boots", "sandals", "no_shoes", "character:cinderella", "footwear_bow", "1girl"];

    private static TagVocabulary Vocabulary()
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        foreach (var tag in Tags)
            vocabulary.AddTag(tag);
        return vocabulary;
    }

    // Danbooru says platform_boots -> boots; the review added boots/sandals -> shoes, footwear_bow implies shoes, no_shoes excludes shoes.
    private static readonly (string, string)[] Danbooru = [("platform_boots", "boots")];

    private static TagParentage Rules() => new(
        [("boots", "shoes"), ("sandals", "shoes")],
        [("footwear_bow", "shoes")],
        [("no_shoes", "shoes")]);

    private static TagFamilies Families() => Rules().Bind(Vocabulary(), Danbooru);

    [Fact]
    public void Load_KeepsAcceptedRulesByKind_AndStripsPrefixes()
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.json");
        File.WriteAllText(path, """
            {"rules": [
              {"child": "boots", "parent": "shoes", "kind": "is_a", "draft": "accept", "verdict": "accept"},
              {"child": "drawing_bow", "parent": "bow", "kind": "is_a", "draft": "reject", "verdict": null},
              {"child": "tabi", "parent": "shoes", "kind": "is_a", "draft": "accept", "verdict": "unsure"},
              {"child": "geta", "parent": "shoes", "kind": "is_a", "draft": "accept", "verdict": null},
              {"child": "bridal_gauntlets", "parent": "gauntlets", "kind": "excludes", "draft": "reject", "verdict": "accept"},
              {"child": "series:x", "parent": "crown", "kind": "implies", "draft": "accept", "verdict": "accept"}
            ]}
            """);
        try
        {
            var rules = TagParentage.Load(path);
            Assert.Equal([("boots", "shoes"), ("geta", "shoes")], rules.IsA);
            Assert.Equal([("bridal_gauntlets", "gauntlets")], rules.Excludes);
            Assert.Equal([("x", "crown")], rules.Implies);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ResolvePath_NoneTurnsRulesOff_NullUsesTheSavedCopy()
    {
        Assert.Null(TagParentage.ResolvePath("none", "/ckpt"));
        Assert.Equal(Path.Combine("/ckpt", TagParentage.FileName), TagParentage.ResolvePath(null, "/ckpt"));
        Assert.Equal("/rules.json", TagParentage.ResolvePath("/rules.json", "/ckpt"));
    }

    [Fact]
    public void Completion_ChainsThroughDanbooruAndReviewedRules()
    {
        var implications = TagImplications.FromRawPairs(Danbooru.Concat(Rules().CompletionPairs), Vocabulary());
        var completed = implications.Complete([[2], [6], [4]], out _, out _);
        Assert.Equal([0, 1, 2], completed[0].Order()); // platform_boots -> boots -> shoes
        Assert.Equal([0, 6], completed[1].Order());    // footwear_bow implies shoes
        Assert.Equal([4], completed[2]);               // excludes never completes
    }

    [Fact]
    public void FamilyAncestors_ComeOnlyThroughAReviewedRule()
    {
        var families = Families();
        Assert.Equal([0], families.FamilyAncestors(1));  // boots: shoes
        Assert.Equal([0], families.FamilyAncestors(2));  // platform_boots: shoes, not boots (Danbooru's own rule)
        Assert.Empty(families.FamilyAncestors(6));       // footwear_bow only implies shoes
        Assert.Equal([1, 2, 3], families.Kinds(0));
        Assert.Equal([0, 1], families.KindOf(2));
    }

    [Fact]
    public void IsUnknown_OnlyWhenTheParentIsTaggedWithNoKindNamed()
    {
        var families = Families();
        Assert.True(families.IsUnknown([0, 7], 1));          // shoes, which? boots unknown
        Assert.False(families.IsUnknown([0, 3, 7], 1));      // shoes + sandals: boots is an ordinary negative
        Assert.False(families.IsUnknown([7], 1));            // no shoes tagged at all
        Assert.False(families.IsUnknown([0, 1], 2));         // boots, which boots? Danbooru's meaning: ordinary negative
        Assert.True(families.IsUnknown([0, 6], 1));          // a footwear bow doesn't name the kind
        Assert.Equal([1, 2, 3], families.UnknownRows([0, 7]).Order());
        Assert.Empty(families.UnknownRows([0, 3]));
    }

    [Fact]
    public void ExcludedRows_RuleOutTheParentAndEveryKind_UnlessTheParentIsTagged()
    {
        var families = Families();
        Assert.Equal([0, 1, 2, 3], families.ExcludedRows([4, 7]).Order());
        Assert.Empty(families.ExcludedRows([4, 0, 3]));     // no_shoes and shoes: a second character's
        Assert.Equal([4], families.Excluders(1));
    }

    [Fact]
    public void Statistics_NeverUseAFamilyAncestorAsAHardNegativeSource_AndPutExcludersFirst()
    {
        var images = new List<IReadOnlyList<int>>();
        for (var i = 0; i < 40; i++)
            images.Add([0, 1, 7]);  // boots (completed shoes)
        for (var i = 0; i < 40; i++)
            images.Add([0, 7]);     // shoes, kind not named
        for (var i = 0; i < 20; i++)
            images.Add([4, 7]);     // no_shoes
        var families = Families();

        var plain = TagStatistics.Build(images, Tags.Length, samplesPerTag: 1000);
        Assert.Contains(0, plain.HardNegativeSources[1]);

        var statistics = TagStatistics.Build(images, Tags.Length, samplesPerTag: 1000, families: families);
        Assert.DoesNotContain(0, statistics.HardNegativeSources[1]);
        Assert.Equal(4, statistics.HardNegativeSources[1][0]);
        Assert.Equal(4, statistics.HardNegativeSources[0][0]);

        // A shoes-only image is never sampled as a "no boots".
        var random = new Random(1);
        for (var i = 0; i < 200; i++)
            if (statistics.SampleHardNegative(1, random) is int image)
                Assert.DoesNotContain(0, images[image]);
    }

    [Fact]
    public void PairWeights_UnnamedKindsGetNoWeight_ExcludedOnesAreCertain()
    {
        var images = new List<IReadOnlyList<int>>();
        for (var i = 0; i < 10; i++)
            images.Add([0, 1, 2, 3, 4, 7]);
        var statistics = TagStatistics.Build(images, Tags.Length, samplesPerTag: 1000, impliedThreshold: 1.1, families: Families());
        var builder = new PairWeightBuilder(statistics, ExclusiveTagGroups.Empty, certainNegativeWeight: 2f);

        using var targets = builder.Build([[0, 7], [0, 3, 7], [4, 7], [4, 0, 7]]);
        var weights = targets.Weights.data<float>().ToArray();
        var focal = targets.FocalMask.data<float>().ToArray();
        float W(int image, int tag) => weights[image * Tags.Length + tag];
        float F(int image, int tag) => focal[image * Tags.Length + tag];

        Assert.Equal((0f, 0f, 0f), (W(0, 1), W(0, 2), W(0, 3)));         // shoes, which? unknown
        Assert.Equal((1f, 1f), (W(1, 1), F(1, 1)));                        // shoes + sandals: boots unverified
        Assert.Equal((2f, 0f), (W(2, 0), F(2, 0)));                        // no_shoes: certainly no shoes...
        Assert.Equal((2f, 0f), (W(2, 1), F(2, 1)));                        // ...nor boots
        Assert.Equal(0f, W(3, 1));                                         // no_shoes + shoes: the rule stands aside
    }

    [Fact]
    public void ExcluderAnchors_PairTheRuledOutTagWithExcluderImages_WeightedAsSet()
    {
        var images = new List<IReadOnlyList<int>>();
        for (var i = 0; i < 20; i++)
            images.Add([0, 1, 7]);  // boots (completed shoes)
        for (var i = 0; i < 20; i++)
            images.Add([4, 7]);     // no_shoes
        images.Add([4, 0, 7]);      // no_shoes and shoes: a second character's
        var statistics = TagStatistics.Build(images, Tags.Length, samplesPerTag: 1000, families: Families());
        var sampler = new TagCentricBatchSampler(statistics, 2, 0, random: new Random(2)) { ExcluderShare = 1 };

        var batch = Enumerable.Range(0, 50).SelectMany(_ => sampler.SampleBatch(4)).ToList();
        var negatives = batch.Where(s => s.IsHardNegative).ToList();
        Assert.NotEmpty(negatives);
        Assert.All(negatives, s => Assert.InRange(s.Image, 20, 39));         // a no_shoes image never tagged shoes
        Assert.All(batch, s => Assert.Equal(0, s.AnchorTag)); // the ruled-out parent, shoes, not its kinds

        var builder = new PairWeightBuilder(statistics, ExclusiveTagGroups.Empty, certainNegativeWeight: 1f, excludedNegativeWeight: 3f);
        using var targets = builder.Build([images[20]]);
        Assert.Equal(3f, targets.Weights.data<float>()[0]);
    }

    [Fact]
    public void ExpandFamilies_KindBoxesCountForTheParent_ParentOnlyBoxesAreIgnoredForKinds()
    {
        var content = new LetterboxBox(0, 0, 512, 512);
        GoldImage[] gold =
        [
            new(1, content, ["boots"], [new GoldInstance("boots", [10, 10, 20, 20], GoldKind.Certain)], new HashSet<string>(), "clean"),
            new(2, content, ["boots", "shoes"],
                [new GoldInstance("boots", [10, 10, 20, 20], GoldKind.Certain), new GoldInstance("shoes", [11, 10, 20, 20], GoldKind.Certain),
                 new GoldInstance("shoes", [200, 200, 20, 20], GoldKind.Certain)],
                new HashSet<string>(), "clean"),
        ];

        var expanded = LocalizationEvaluator.ExpandFamilies(gold, Families(), Vocabulary());

        Assert.Equal(["boots", "shoes"], expanded[0].Targets);
        Assert.Contains(expanded[0].Instances, x => x is { Tag: "shoes", Kind: GoldKind.Certain });

        // The shoes box over the boots isn't doubled; the other shoes box (kind not named) is ignored for boots.
        Assert.Equal(2, expanded[1].Instances.Count(x => x.Tag == "shoes"));
        Assert.Single(expanded[1].Instances, x => x is { Tag: "boots", Kind: GoldKind.Ignore } && x.Box[0] == 200);
    }
}
