using UnbooruTagger.Core.Scoring;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class ContextNegativeTests
{
    // Tag 0 = a character, 1 = hat (on 30 of her 40 images: P = 0.75), 2 = her hair color
    // (on all 40: P = 1), 3 = unrelated. 20 more images have a hat without her.
    private static List<IReadOnlyList<int>> Corpus()
    {
        var images = new List<IReadOnlyList<int>>();
        for (var i = 0; i < 30; i++)
            images.Add([0, 1, 2]);
        for (var i = 0; i < 10; i++)
            images.Add([0, 2]);     // her without the hat: real absences
        for (var i = 0; i < 20; i++)
            images.Add([1]);
        for (var i = 0; i < 60; i++)
            images.Add([3]);
        return images;
    }

    private static TagStatistics Statistics(double? missingThreshold = 0.9) =>
        TagStatistics.Build(Corpus(), 4, samplesPerTag: 1000, impliedThreshold: 0.6, hardNegativeMinCounterExamples: 5, missingThreshold: missingThreshold);

    [Fact]
    public void Build_SplitsNearCertainImplicationsFromContext()
    {
        var statistics = Statistics();
        Assert.Contains(2, statistics.ImpliedBy[0]);        // hair color: P = 1, a missing label when absent
        Assert.DoesNotContain(1, statistics.ImpliedBy[0]);   // hat: P = 0.75, not "missing"...
        Assert.Contains(0, statistics.ContextSources[1]);    // ...but context for the hat
        Assert.False(statistics.IsLikelyMissing(30, 1));     // her, hatless
        Assert.True(statistics.IsContextNegative(30, 1));

        var original = Statistics(missingThreshold: null);   // no band: the original behavior
        Assert.Contains(1, original.ImpliedBy[0]);
        Assert.Empty(original.ContextSources[1]);
    }

    [Fact]
    public void Sampling_DrawsContextNegatives_AndKeepsThemOutOfHardNegatives()
    {
        var statistics = Statistics();
        var random = new Random(2);
        for (var i = 0; i < 200; i++)
        {
            Assert.InRange(statistics.SampleContextNegative(1, random)!.Value, 30, 39);
            var hard = statistics.SampleHardNegative(1, random);
            Assert.True(hard is null || !statistics.IsContextNegative(hard.Value, 1));
        }

        var sampler = new TagCentricBatchSampler(statistics, 2, 2, random: new Random(3), contextNegativesPerAnchor: 1);
        var batch = sampler.SampleBatch(500);
        Assert.Contains(batch, s => s.IsContextNegative);
        foreach (var s in batch.Where(s => s.IsContextNegative))
            Assert.False(statistics.HasTag(s.Image, s.AnchorTag));
    }

    [Fact]
    public void Weights_TrainContextNegativesAsUnverified_InsteadOfIgnoringThem()
    {
        var builder = new PairWeightBuilder(Statistics(), ExclusiveTagGroups.Empty);
        using var targets = builder.Build([[0, 2]], [new SampledImage(30, AnchorTag: 1, IsHardNegative: false, IsContextNegative: true)]);
        var (labels, weights, focal) = (targets.Labels.data<float>().ToArray(), targets.Weights.data<float>().ToArray(), targets.FocalMask.data<float>().ToArray());
        Assert.Equal((-1f, 1f, 1f), (labels[1], weights[1], focal[1])); // hat: an unverified (focal) "no"
        Assert.Equal(1f, labels[2]);                                     // hair color: positive

        var original = new PairWeightBuilder(Statistics(missingThreshold: null), ExclusiveTagGroups.Empty);
        using var originalTargets = original.Build([[0, 2]]);
        Assert.Equal(0f, originalTargets.Weights.data<float>().ToArray()[1]); // before: ignored as likely missing
    }

    [Fact]
    public void Weights_WithAContextNegativeWeight_TrainEveryContextNegativeAtFullStrength()
    {
        var builder = new PairWeightBuilder(Statistics(), ExclusiveTagGroups.Empty, contextNegativeWeight: 2f);
        // Her without the hat (not sampled as anyone's context negative), her with it, and someone else.
        using var targets = builder.Build([[0, 2], [0, 1, 2], [3]]);
        var (labels, weights, focal) = (targets.Labels.data<float>().ToArray(), targets.Weights.data<float>().ToArray(), targets.FocalMask.data<float>().ToArray());
        Assert.Equal((-1f, 2f, 0f), (labels[1], weights[1], focal[1]));   // hatless her: a full-strength "no"
        Assert.Equal((1f, 1f, 0f), (labels[5], weights[5], focal[5]));    // her with the hat: still a positive
        Assert.Equal((-1f, 1f, 1f), (labels[9], weights[9], focal[9]));   // no context: an unverified "no"
        using var hairless = builder.Build([[0]]);
        Assert.Equal(0f, hairless.Weights.data<float>().ToArray()[2]);    // her without her hair color: still likely missing
    }

    [Fact]
    public void Weights_LeaveContextNegativesOfIneligibleTagsUnverified()
    {
        // Hat treated as a whole-image tag (like 1girl): its context negatives stay focal-damped.
        var builder = new PairWeightBuilder(Statistics(), ExclusiveTagGroups.Empty, contextNegativeWeight: 2f, contextNegativeTags: [true, false, true, true]);
        using var targets = builder.Build([[0, 2]]);
        Assert.Equal((-1f, 1f, 1f), (targets.Labels.data<float>().ToArray()[1], targets.Weights.data<float>().ToArray()[1], targets.FocalMask.data<float>().ToArray()[1]));
    }

    [Fact]
    public void Evaluate_ContextAp_CatchesAClassifierThatAnswersFromContext()
    {
        // The hat score follows the character, not the hat: hatless images of her score as
        // high as real hats (2-d embeddings: dimension 0 = "hat", dimension 1 = anything else).
        var images = Corpus();
        var pooled = images.SelectMany(tags => tags.Contains(1) || tags.Contains(0) ? new[] { 1f, 0f } : [0f, 1f]).ToArray();
        var vocabulary = TagVocabulary.CreateEmpty();
        foreach (var tag in new[] { "character", "hat", "hair", "other" })
            vocabulary.AddTag(tag);
        var embeddings = new[] { 1f, 0f, 1f, 0f, 1f, 0f, 0f, 1f };

        var report = TagRankingEvaluator.Evaluate(pooled, images, embeddings, 2, new LogitScaling(10f, -5f), vocabulary, Statistics(),
            ExclusiveTagGroups.Empty, CPU, minPositives: 5);
        var hat = report.Tags.Single(t => t.Tag == "hat");

        Assert.Equal(10, hat.ContextNegatives);
        Assert.True(hat.AveragePrecision > 0.9);          // looks fine against hard + random negatives
        Assert.True(hat.ContextAveragePrecision < 0.9);   // but can't tell the hat from her
        Assert.False(double.IsNaN(report.MeanContextAveragePrecision));
    }

    [Fact]
    public void TeacherGate_RejectsATagThatFailsItsContextNegatives()
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        vocabulary.AddTag("hat");
        var report = new EvaluationReport(100, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            [new TagEvaluation("hat", 50, 10, 10, 0.95, 0.9, 0.9, 0.5f, 0.9, ContextNegatives: 40, ContextAveragePrecision: 0.5)]);
        Assert.Equal(0, CropTeacherGate.From(report, vocabulary, [true], 0.8, 20).TrustedCount);

        var honest = report with { Tags = [report.Tags[0] with { ContextAveragePrecision = 0.85 }] };
        Assert.Equal(1, CropTeacherGate.From(honest, vocabulary, [true], 0.8, 20).TrustedCount);
    }

    [Fact]
    public void Build_WithTagRecall_DecidesMissingLabelsPerTag()
    {
        TagStatistics With(double hatRecall) => TagStatistics.Build(Corpus(), 4, samplesPerTag: 1000, impliedThreshold: 0.6,
            hardNegativeMinCounterExamples: 5, missingThreshold: 0.9, tagRecall: [double.NaN, hatRecall, 0.9, double.NaN]);

        // Reliably tagged hat (r = 0.9): P(hat there | untagged on her) = 0.075 / 0.325 = 0.23 -> a real absence.
        var reliable = With(0.9);
        Assert.DoesNotContain(1, reliable.ImpliedBy[0]);
        Assert.Contains(0, reliable.ContextSources[1]);
        Assert.Contains(2, reliable.ImpliedBy[0]); // her hair color (p = 1) is always "missing" when absent

        // Sloppily tagged hat (r = 0.3): 0.525 / 0.775 = 0.68 -> probably just not tagged.
        var sloppy = With(0.3);
        Assert.Contains(1, sloppy.ImpliedBy[0]);
        Assert.DoesNotContain(0, sloppy.ContextSources[1]);
    }
}
