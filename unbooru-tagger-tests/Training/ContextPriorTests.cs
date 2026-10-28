using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class ContextPriorTests
{
    // Tag 0 = a character, 1 = gloves (on 45 of her 50 images: P = 0.9), 2 = an artist who draws gloves
    // on 20 of 50 (P = 0.4), 3 = sword (on 2 of her images: too few to count), 4 = a whole-image tag
    // (on all her images, not eligible). 400 more images: 40 with gloves, the rest a filler tag 5.
    private static List<IReadOnlyList<int>> Corpus()
    {
        var images = new List<IReadOnlyList<int>>();
        for (var i = 0; i < 50; i++)
            images.Add(i < 45 ? (i < 2 ? [0, 1, 3, 4] : [0, 1, 4]) : [0, 4]);
        for (var i = 0; i < 50; i++)
            images.Add(i < 20 ? [2, 1] : [2]);
        for (var i = 0; i < 400; i++)
            images.Add(i < 40 ? [1] : [5]);
        return images;
    }

    private static readonly bool[] IsSource = [true, false, true, false, false, false];
    private static readonly bool[] Eligible = [false, true, false, true, false, true];

    private static ContextPrior Prior(float weight = 1f) =>
        ContextPrior.Build(TagStatistics.Build(Corpus(), 6, samplesPerTag: 1000), IsSource, Eligible, weight, minSourceImages: 10);

    private static float Logit(double p) => (float)Math.Log(p / (1 - p));

    [Fact]
    public void Offset_IsThePriorsLogOddsOverTheBaseRate_ForEligibleTagsOnly()
    {
        var prior = Prior();
        var row = new float[6];
        prior.Fill([0, 4], row);

        // gloves: P(gloves | her) = 0.9 against a base rate of 105 / 500.
        Assert.Equal(Logit(0.9) - Logit(105 / 500.0), row[1], 4);
        Assert.Equal(0f, row[3]); // 2 sampled co-occurrences: too few
        Assert.Equal(0f, row[4]); // not eligible (whole-image)
        Assert.Equal(0f, row[5]);
    }

    [Fact]
    public void Fill_TakesTheHighestOverTheImagesSources_AndScalesWithTheWeight()
    {
        var both = new float[6];
        Prior().Fill([2, 0], both);
        var artistOnly = new float[6];
        Prior().Fill([2], artistOnly);
        Assert.Equal(Logit(0.4) - Logit(105 / 500.0), artistOnly[1], 4);
        Assert.Equal(Logit(0.9) - Logit(105 / 500.0), both[1], 4);

        var half = new float[6];
        Prior(0.5f).Fill([0], half);
        Assert.Equal(both[1] / 2, half[1], 4);

        var none = new float[6];
        Prior().Fill([5], none);
        Assert.All(none, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void PairWeightBuilder_CarriesTheOffsets_AndTheLossAddsThemToTheLogits()
    {
        var statistics = TagStatistics.Build(Corpus(), 6, samplesPerTag: 1000);
        var prior = Prior();
        using var targets = new PairWeightBuilder(statistics, ExclusiveTagGroups.Empty, contextPrior: prior).Build([[0, 4], [5]]);
        Assert.NotNull(targets.LogitOffsets);
        var offsets = targets.LogitOffsets!.data<float>().ToArray();
        Assert.True(offsets[1] > 0);       // her image, gloves
        Assert.Equal(0f, offsets[6 + 1]);  // the filler image

        // The offset acts exactly like a shifted logit.
        var image = tensor(new[] { 1f, 0f }, [1, 2]);
        var tags = tensor(new[] { 0.5f, 0f, -0.3f, 0f }, [2, 2]);
        using var plain = new PairTargets(tensor(new[] { 1f, -1f }, [1, 2]), ones(1, 2), zeros(1, 2), 1f);
        var shifted = plain with { LogitOffsets = tensor(new[] { 2f, 1f }, [1, 2]) };
        using var moved = tensor(new[] { 2.5f, 0f, 0.7f, 0f }, [2, 2]);
        Assert.Equal(
            SigmoidContrastiveLoss.ComputeWeighted(image, moved, plain, null, 4f, 0.05f).item<float>(),
            SigmoidContrastiveLoss.ComputeWeighted(image, tags, shifted, null, 4f, 0.05f).item<float>(), 5);
    }
}
