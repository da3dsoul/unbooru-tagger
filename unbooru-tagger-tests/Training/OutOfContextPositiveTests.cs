using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class OutOfContextPositiveTests
{
    // Tag 0 = a character who wears gloves (tag 1) on 45 of her 50 images; 40 other images have gloves and no
    // character (prior 0); tag 2 = a filler tag on 400 images.
    private static List<IReadOnlyList<int>> Corpus()
    {
        var images = new List<IReadOnlyList<int>>();
        for (var i = 0; i < 50; i++)
            images.Add(i < 45 ? [0, 1] : [0]);
        for (var i = 0; i < 40; i++)
            images.Add([1]);
        for (var i = 0; i < 400; i++)
            images.Add([2]);
        return images;
    }

    private static readonly bool[] IsSource = [true, false, false];
    private static readonly bool[] Eligible = [false, true, true];

    [Fact]
    public void Prior_IsTheHighestOverTheImagesSources()
    {
        var statistics = TagStatistics.Build(Corpus(), 3, samplesPerTag: 1000);
        var table = ContextPriorTable.Build(statistics, IsSource, Eligible, minSourceImages: 10);
        Assert.Equal(0.9f, table.Prior([0, 1], 1), 4);
        Assert.Equal(0f, table.Prior([1], 1));
        Assert.Equal(0f, table.Prior([0, 1], 0)); // not eligible
    }

    [Fact]
    public void Sampler_PicksLowPriorPositives_AndTheirAnchorPairGetsTheWeight()
    {
        var corpus = Corpus();
        var statistics = TagStatistics.Build(corpus, 3, samplesPerTag: 1000);
        var table = ContextPriorTable.Build(statistics, IsSource, Eligible, minSourceImages: 10);
        var sampler = new TagCentricBatchSampler(statistics, 2, 0, random: new Random(5))
        {
            OutOfContextPriors = table, OutOfContextTags = Eligible, OutOfContextShare = 1, OutOfContextMaxPrior = 0.3,
        };

        var gloves = Enumerable.Range(0, 200).SelectMany(_ => sampler.SampleBatch(4)).Where(s => s.AnchorTag == 1).ToList();
        Assert.NotEmpty(gloves);
        Assert.All(gloves, s => Assert.True(s.IsOutOfContextPositive && s.Image >= 50)); // never the character's images
        Assert.DoesNotContain(Enumerable.Range(0, 50).SelectMany(_ => sampler.SampleBatch(4)), s => s.AnchorTag == 0 && s.IsOutOfContextPositive);

        var batch = new[] { new SampledImage(60, 1, false, IsOutOfContextPositive: true), new SampledImage(0, 1, false) };
        var builder = new PairWeightBuilder(statistics, ExclusiveTagGroups.Empty, outOfContextPositiveWeight: 3f);
        using var targets = builder.Build([corpus[60], corpus[0]], batch);
        var weights = targets.Weights.data<float>().ToArray();
        Assert.Equal(3f, weights[0 * 3 + 1]);
        Assert.Equal(1f, weights[1 * 3 + 1]);

        // With the table, every low-prior positive in the batch, sampled as an anchor or not.
        var everywhere = new PairWeightBuilder(statistics, ExclusiveTagGroups.Empty, outOfContextPositiveWeight: 3f,
            outOfContextPriors: table, outOfContextTags: Eligible, outOfContextMaxPrior: 0.3f);
        using var all = everywhere.Build([corpus[60], corpus[0]]);
        var allWeights = all.Weights.data<float>().ToArray();
        Assert.Equal(3f, allWeights[0 * 3 + 1]); // gloves, no character: prior 0
        Assert.Equal(1f, allWeights[1 * 3 + 1]); // her gloves: prior 0.9
    }
}
