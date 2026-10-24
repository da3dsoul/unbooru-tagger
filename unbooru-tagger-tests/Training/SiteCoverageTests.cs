using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Scoring;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class SiteCoverageTests
{
    private const BooruSites Both = BooruSites.Danbooru | BooruSites.Gelbooru;

    [Fact]
    public void LoadSiteCoverage_FindsSingleSiteTags_AndAssumesBothWithoutEnoughOverlap()
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        foreach (var tag in new[] { "hat", "silver_hair", "covered_nipples", "rare" })
            vocabulary.AddTag(tag);
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.json");
        File.WriteAllText(path, """
            {"Tags": {"hat": [1000, 30, 10], "silver_hair": [0, 0, 500], "covered_nipples": [2, 400, 0], "rare": [0, 0, 20]}}
            """);
        try
        {
            var sites = TagAgreement.LoadSiteCoverage(path, vocabulary)!;
            Assert.Equal([Both, BooruSites.Gelbooru, BooruSites.Danbooru, Both], sites);
        }
        finally
        {
            File.Delete(path);
        }
    }

    // Tag 0 = silver_hair (Gelbooru-only name), tag 1 = anything. Images 0-9 are Gelbooru
    // images with silver_hair; 10-29 Danbooru-only images without it (some really are silver-haired).
    private static TagStatistics Statistics()
    {
        var rows = new List<IReadOnlyList<int>>();
        var sites = new List<BooruSites>();
        for (var i = 0; i < 10; i++) { rows.Add([0, 1]); sites.Add(BooruSites.Gelbooru); }
        for (var i = 0; i < 20; i++) { rows.Add([1]); sites.Add(BooruSites.Danbooru); }
        for (var i = 0; i < 20; i++) { rows.Add([1]); sites.Add(Both); }
        return TagStatistics.Build(rows, 2, samplesPerTag: 1000, imageSites: sites, tagSites: [BooruSites.Gelbooru, Both]);
    }

    [Fact]
    public void SiteBlindAbsences_AreIgnoredInTraining_AndNeverSampledAsNegatives()
    {
        var statistics = Statistics();
        Assert.True(statistics.IsSiteBlind(10, 0));   // Danbooru-only image, Gelbooru-only tag
        Assert.False(statistics.IsSiteBlind(30, 0));  // on both sites: a real absence
        Assert.False(statistics.IsSiteBlind(10, 1));

        var random = new Random(4);
        for (var i = 0; i < 300; i++)
            Assert.False(statistics.IsSiteBlind(statistics.SampleHardNegative(0, random)!.Value, 0));

        var builder = new PairWeightBuilder(statistics, ExclusiveTagGroups.Empty);
        using var targets = builder.Build([[1], [1], [0, 1]], batchSites: [BooruSites.Danbooru, Both, BooruSites.Gelbooru]);
        var weights = targets.Weights.data<float>().ToArray();
        Assert.Equal(0f, weights[0]);      // Danbooru-only image: silver_hair unknown
        Assert.Equal(1f, weights[2]);      // image on both sites: a real (unverified) negative
        Assert.Equal(1f, weights[4]);      // carries it: positive
    }

    [Fact]
    public void Evaluate_DoesNotCountSiteBlindImagesAsNegatives()
    {
        // The classifier (rightly) scores the silver-haired Danbooru-only images high; with
        // site coverage they aren't negatives, so AP stays perfect.
        var statistics = Statistics();
        var heldTags = new List<IReadOnlyList<int>>();
        var heldSites = new List<BooruSites>();
        var pooled = new List<float>();
        for (var i = 0; i < 10; i++) { heldTags.Add([0, 1]); heldSites.Add(BooruSites.Gelbooru); pooled.AddRange([1f, 0f]); }
        for (var i = 0; i < 10; i++) { heldTags.Add([1]); heldSites.Add(BooruSites.Danbooru); pooled.AddRange([1f, 0f]); }
        for (var i = 0; i < 40; i++) { heldTags.Add([1]); heldSites.Add(Both); pooled.AddRange([0f, 1f]); }
        var vocabulary = TagVocabulary.CreateEmpty();
        vocabulary.AddTag("silver_hair");
        vocabulary.AddTag("other");

        EvaluationReport Run(IReadOnlyList<BooruSites>? sites) => TagRankingEvaluator.Evaluate(
            pooled.ToArray(), heldTags, [1f, 0f, 0f, 1f], 2, new LogitScaling(10f, -5f), vocabulary, statistics,
            ExclusiveTagGroups.Empty, CPU, minPositives: 5, minCalibrationPositives: 5, imageSites: sites);

        var aware = Run(heldSites).Tags.Single(t => t.Tag == "silver_hair");
        var blind = Run(null).Tags.Single(t => t.Tag == "silver_hair");
        Assert.Equal(1.0, aware.AveragePrecision, 6);
        Assert.True(blind.AveragePrecision < 1.0);
        Assert.Equal(1.0, Run(heldSites).MicroPrecision, 6);
    }
}
