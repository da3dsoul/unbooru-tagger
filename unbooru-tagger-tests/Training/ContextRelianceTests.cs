using UnbooruTagger.Core.Scoring;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class ContextRelianceTests
{
    // 0 = a character who wears gloves on 80% of her training images, 1 = gloves, 2 = anything else.
    private static TagStatistics TrainingStatistics()
    {
        var images = new List<IReadOnlyList<int>>();
        for (var i = 0; i < 80; i++)
            images.Add([0, 1]);
        for (var i = 0; i < 20; i++)
            images.Add([0]);
        for (var i = 0; i < 100; i++)
            images.Add([1, 2]);
        for (var i = 0; i < 300; i++)
            images.Add([2]);
        return TagStatistics.Build(images, 3, samplesPerTag: 1000, hardNegativeMinCounterExamples: 5, missingThreshold: 0.9);
    }

    // Held out: her with and without gloves, others with and without gloves.
    private static readonly IReadOnlyList<int>[] HeldOut =
    [
        .. Enumerable.Repeat<IReadOnlyList<int>>([0, 1], 20),
        .. Enumerable.Repeat<IReadOnlyList<int>>([0], 20),
        .. Enumerable.Repeat<IReadOnlyList<int>>([1, 2], 20),
        .. Enumerable.Repeat<IReadOnlyList<int>>([2], 40),
    ];

    private static ContextRelianceReport Run(Func<IReadOnlyList<int>, bool> glovesScoreHigh)
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        foreach (var tag in new[] { "character:her", "gloves", "other" })
            vocabulary.AddTag(tag);
        // 2-d embeddings: dimension 0 is what the "gloves" row reads.
        var pooled = HeldOut.SelectMany(tags => glovesScoreHigh(tags) ? new[] { 1f, 0f } : [0f, 1f]).ToArray();
        float[] embeddings = [1f, 0f, 1f, 0f, 0f, 1f];
        return ContextRelianceEvaluator.Evaluate(pooled, HeldOut, embeddings, 2, new LogitScaling(10f, -5f), vocabulary,
            TrainingStatistics(), [0.5f, 0.5f, 0.5f], CPU);
    }

    [Fact]
    public void AClassifierThatSeesTheObject_HasNoContextGap()
    {
        var gloves = Run(tags => tags.Contains(1)).Tags.Single(t => t.Tag == "gloves");

        Assert.Equal(0, gloves.RecallGap, 6);
        Assert.Equal(0, gloves.FalsePositiveGap, 6);
    }

    [Fact]
    public void AClassifierThatAnswersFromTheCharacter_ShowsBothGaps()
    {
        var report = Run(tags => tags.Contains(0));
        var gloves = report.Tags.Single(t => t.Tag == "gloves");

        Assert.Equal(1, gloves.RecallGap, 6);         // misses every pair of gloves on someone else
        Assert.Equal(1, gloves.FalsePositiveGap, 6);  // and calls gloves on every image of her without them
        Assert.Equal(1, report.MacroFalsePositiveGap, 6);

        var usual = report.Bands.Single(b => b.Name == "prior 60-90%");
        Assert.Equal((20, 20), (usual.Positives, usual.Negatives)); // her images land in her 80% band
        Assert.Equal(1, usual.FalsePositiveRate, 6);
    }

    [Fact]
    public void CharacterTags_AreContextNotMeasuredTags()
    {
        Assert.DoesNotContain(Run(tags => tags.Contains(1)).Tags, t => t.Tag.StartsWith("character:"));
        Assert.True(ContextRelianceEvaluator.IsContextSource("artist:someone"));
        Assert.True(ContextRelianceEvaluator.IsContextSource("series:something"));
        Assert.False(ContextRelianceEvaluator.IsGeneralTag("meta:highres"));
    }

    [Theory]
    [InlineData(0.0, 0)]
    [InlineData(0.049, 0)]
    [InlineData(0.05, 1)]
    [InlineData(0.3, 2)]
    [InlineData(0.6, 3)]
    [InlineData(0.89, 3)]
    [InlineData(0.9, 4)]
    [InlineData(1.0, 4)]
    public void Band_UsesLowerEdges(double prior, int band) => Assert.Equal(band, ContextRelianceEvaluator.Band(prior));

    [Fact]
    public void NegativesAtNearCertainPriors_DontCountTowardTheFalsePositiveGap()
    {
        // Every negative in the >= 90% band is called, but those are mostly missing labels:
        // with nothing in the 60-90% band there is no gap to report.
        var counts = new int[ContextRelianceEvaluator.BandEdges.Length * 4];
        counts[2] = 50;      // < 5%: 50 negatives, none called
        counts[4 * 4 + 2] = 50;
        counts[4 * 4 + 3] = 50;
        var tag = ContextRelianceEvaluator.Summarize([("gloves", counts)], minPerSide: 10);

        Assert.Empty(tag.Tags);
        Assert.True(double.IsNaN(tag.MacroFalsePositiveGap));
        Assert.Equal(1, tag.Bands[4].FalsePositiveRate, 6); // still shown in its own band
    }
}

public class OcclusionGeometryTests
{
    [Fact]
    public void ControlRectangles_KeepSizes_StayInside_AndAvoidTheObject()
    {
        int[] content = [10, 20, 200, 150];
        List<int[]> objects = [[30, 40, 20, 30], [100, 100, 40, 10]];
        var random = new Random(1);
        for (var trial = 0; trial < 50; trial++)
        {
            var control = LocalizationEvaluator.ControlRectangles(objects, content, random)!;
            Assert.Equal(objects.Select(o => (o[2], o[3])).OrderBy(s => s), control.Select(c => (c[2], c[3])).OrderBy(s => s));
            foreach (var c in control)
            {
                Assert.InRange(c[0], content[0], content[0] + content[2] - c[2]);
                Assert.InRange(c[1], content[1], content[1] + content[3] - c[3]);
                Assert.All(objects, o => Assert.False(Intersects(c, o)));
            }
            Assert.False(Intersects(control[0], control[1]));
        }
    }

    [Fact]
    public void ControlRectangles_GiveUpWhenTheObjectFillsTheImage()
    {
        Assert.Null(LocalizationEvaluator.ControlRectangles([[0, 0, 80, 100]], [0, 0, 100, 100], new Random(0)));
    }

    [Fact]
    public void Grow_AddsTheMarginAndClipsToContent()
    {
        Assert.Equal([6, 16, 18, 18], LocalizationEvaluator.Grow([10, 20, 10, 10], 4, [0, 16, 100, 100]));
        Assert.Equal([0, 0, 14, 14], LocalizationEvaluator.Grow([0, 0, 10, 10], 4, [0, 0, 100, 100]));
    }

    [Fact]
    public void Blank_ClearsOnlyTheRectangle_InEveryChannel()
    {
        const int size = 8;
        var image = Enumerable.Repeat(1f, 3 * size * size).ToArray();
        LocalizationEvaluator.Blank(image, [[2, 3, 3, 2], [6, 6, 5, 5]], size); // the second runs off the edge

        for (var ch = 0; ch < 3; ch++)
            for (var y = 0; y < size; y++)
                for (var x = 0; x < size; x++)
                {
                    var inside = (x is >= 2 and < 5 && y is >= 3 and < 5) || (x >= 6 && y >= 6);
                    Assert.Equal(inside ? 0f : 1f, image[ch * size * size + y * size + x]);
                }
    }

    private static bool Intersects(int[] a, int[] b) =>
        a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3];
}
