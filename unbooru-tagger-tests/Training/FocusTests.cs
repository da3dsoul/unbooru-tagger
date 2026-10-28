using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class FocusTests
{
    private static ContextBand Band(double minPrior, int positives, int negatives) => new($"{minPrior}", minPrior, positives, 0.5, negatives, 0.1);

    private static ContextTagResult Context(string tag, double recallGap, double falsePositiveGap, int topPositives = 50, int usualNegatives = 50) =>
        new(tag, recallGap, falsePositiveGap, [Band(0, 500, 5000), Band(0.6, topPositives, usualNegatives), Band(0.9, 0, 0)]);

    [Fact]
    public void From_PicksEligibleTagsAnsweredFromContext_ByEitherCheck()
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        foreach (var tag in new[] { "nude", "ribbon", "aqua_hair", "gloves", "hat", "solo", "rare_bow", "glasses" })
            vocabulary.AddTag(tag);
        var report = new EvaluationReport(100, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], ContextReliance: new ContextRelianceReport([], 0, 0, 0, 0,
        [
            Context("nude", 0.05, 0.48),                       // false-positive gap
            Context("aqua_hair", 0.38, 0.1),                   // recall gap
            Context("ribbon", 0.1, 0.1),                       // neither
            Context("solo", 0.4, 0.4),                         // not eligible
            Context("rare_bow", 0.5, 0.5, topPositives: 5, usualNegatives: 5), // too few images to tell
        ]));
        var localization = new LocalizationReport([], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], OcclusionTags:
        [
            new OcclusionResult("gloves", 43, 0.95, 0.014, 0.01, 0.6), // called without the object, barely moves
            new OcclusionResult("hat", 60, 0.95, 0.047, 0.01, 0.8),    // drops enough
            new OcclusionResult("glasses", 5, 1, 0.01, 0.01, 0.5),     // too few pairs
        ]);

        var focus = FocusTags.From(report, localization, vocabulary, [true, true, true, true, true, false, true, true]);
        Assert.Equal(new[] { "nude", "aqua_hair", "gloves" }.Select(t => vocabulary.Records.Single(r => r.Tag == t).RowIndex).Order(), focus.Rows);
        Assert.Empty(FocusTags.From(null, null, vocabulary, [true, true, true, true, true, true, true, true]).Rows);
    }

    // Tag 0 = a character, 1 = hat (on 30 of her 40 images), 2 = her hair color, 3 = unrelated.
    private static TagStatistics Statistics()
    {
        var images = new List<IReadOnlyList<int>>();
        for (var i = 0; i < 30; i++)
            images.Add([0, 1, 2]);
        for (var i = 0; i < 10; i++)
            images.Add([0, 2]);
        for (var i = 0; i < 20; i++)
            images.Add([1]);
        for (var i = 0; i < 60; i++)
            images.Add([3]);
        return TagStatistics.Build(images, 4, samplesPerTag: 1000, impliedThreshold: 0.6, hardNegativeMinCounterExamples: 5, missingThreshold: 0.9);
    }

    [Fact]
    public void Sampler_DrawsFocusAnchors_WithAsManyContextNegativesAsPositives()
    {
        var statistics = Statistics();
        var sampler = new TagCentricBatchSampler(statistics, 2, 2, random: new Random(1), contextNegativesPerAnchor: 1) { FocusShare = 1 };
        sampler.Focus = [1];
        var batch = sampler.SampleBatch(12);
        Assert.All(batch, s => Assert.Equal(1, s.AnchorTag));
        for (var anchor = 0; anchor < batch.Length; anchor += 6)
        {
            Assert.Equal([false, false, true, true], batch.Skip(anchor).Take(4).Select(s => s.IsContextNegative));
            Assert.All(batch.Skip(anchor + 2).Take(2), s => Assert.False(statistics.HasTag(s.Image, 1)));
        }

        // No focus share: anchors stay uniform, with the usual one context negative.
        var plain = new TagCentricBatchSampler(statistics, 2, 2, random: new Random(1), contextNegativesPerAnchor: 1) { Focus = [1] };
        Assert.Contains(plain.SampleBatch(200), s => s.AnchorTag != 1);
    }

    [Fact]
    public void PickTiles_PutsAFocusAnchorsPositivesAndContextNegativesOnOneCanvas()
    {
        SampledImage[] sampled =
        [
            new(0, 3, false), new(1, 3, false), new(2, 3, true),
            new(10, 1, false), new(11, 1, false), new(12, 1, false, IsContextNegative: true), new(13, 1, false, IsContextNegative: true), new(14, 1, true),
            new(5, 3, false), new(6, 3, true),
        ];
        for (var seed = 0; seed < 20; seed++)
        {
            var tiles = Mosaic.PickTiles(10, 2, new Random(seed), sampled, new HashSet<int> { 1 });
            Assert.Equal([3, 4, 5, 6], tiles.Take(4).Order());
            Assert.Equal(8, tiles.Distinct().Count());
        }
    }

    [Fact]
    public void FullSizeTiles_KeepTheImageAtItsSize_OnATwiceAsWideCanvas()
    {
        const int size = 4;
        var pixels = new float[4 * 3 * size * size];
        for (var i = 0; i < 4; i++)
            Array.Fill(pixels, i + 1f, i * 3 * size * size, 3 * size * size);
        pixels[2 * 3 * size * size] = 9f; // image 2's top-left pixel, kept as is
        using var canvas = Mosaic.Compose(tensor(pixels, [4, 3, size, size]), [3, 2, 1, 0], size, tileSize: size);
        Assert.Equal([1L, 3, 2 * size, 2 * size], canvas.shape);
        var values = canvas.data<float>().ToArray();
        Assert.Equal(4f, values[0]);
        Assert.Equal(9f, values[size]);            // top-right tile: image 2
        Assert.Equal(3f, values[size + 1]);
        Assert.Equal(1f, values[2 * size * 2 * size - 1]);

        var content = new LetterboxBox(0, 64, 512, 384);
        Assert.Equal(new LetterboxBox(512, 576, 512, 384), Mosaic.TileBox(content, 3, 512, tileSize: 512));
    }

    [Fact]
    public void TileTargets_AreTheTileImagesOwnRows()
    {
        // 3 images x 2 tags: image 1 has tag 0 (weight 2), image 2 has tag 1.
        using var targets = new PairTargets(
            tensor(new[] { -1f, -1f, 1f, -1f, -1f, 1f }, [3, 2]),
            tensor(new[] { 1f, 1f, 2f, 1f, 1f, 0.5f }, [3, 2]),
            zeros(3, 2), 2.5f);
        using var tile = Mosaic.TileTargets(targets, [2, 1, 1]);
        Assert.Equal([-1f, 1f, 1f, -1f, 1f, -1f], tile.Labels.data<float>().ToArray());
        Assert.Equal(0.5f + 2f + 2f, tile.PositiveWeightSum);
    }
}
