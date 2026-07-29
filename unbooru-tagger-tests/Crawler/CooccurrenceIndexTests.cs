using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

public class TagCooccurrenceIndexTests
{
    private const int Miku = 0;
    private const int Vocaloid = 1;
    private const int OtherVocaloidCharacter = 2;

    private const int Breasts = 0;
    private const int LargeBreasts = 1;

    [Fact]
    public void Build_CountsTagsAndPairsFromImageSets()
    {
        List<HashSet<int>> images =
        [
            [1, 2],
            [1, 2],
            [1, 3],
        ];

        var index = TagCooccurrenceIndex.Build(images);

        // Tag 1 appears in all 3 images; tag 2 co-occurs with it in 2 of those.
        var candidates = index.FindHardNegativeSources(targetTagRow: 1, minCooccurrenceRatio: 0.5, minCounterExamples: 0);
        var tag2 = Assert.Single(candidates, c => c.TagRow == 2);
        Assert.Equal(2, tag2.CooccurrenceCount);
        Assert.Equal(2, tag2.OtherTagImageCount); // tag 2 only ever appears in the 2 images alongside tag 1
        Assert.Equal(0, tag2.CounterExampleCount);
    }

    [Fact]
    public void Build_IgnoresPairsAcrossDifferentImages()
    {
        List<HashSet<int>> images =
        [
            [10],
            [20],
        ];

        var index = TagCooccurrenceIndex.Build(images);

        // Tags 10 and 20 never share an image, so neither should show up as a
        // co-occurrence candidate for the other, regardless of thresholds.
        Assert.Empty(index.FindHardNegativeSources(10, minCooccurrenceRatio: 0, minCounterExamples: 0));
        Assert.Empty(index.FindHardNegativeSources(20, minCooccurrenceRatio: 0, minCounterExamples: 0));
    }

    [Fact]
    public void FindHardNegativeSources_SeriesCharacterCase_Qualifies()
    {
        // 60 hatsune_miku images, every one also tagged vocaloid; 40 more vocaloid
        // images are of some other character entirely (plenty of real counter-examples
        // — art of the series that isn't specifically this character).
        var images = new List<HashSet<int>>();
        images.AddRange(Enumerable.Repeat(0, 60).Select(_ => new HashSet<int> { Miku, Vocaloid }));
        images.AddRange(Enumerable.Repeat(0, 40).Select(_ => new HashSet<int> { Vocaloid, OtherVocaloidCharacter }));

        var index = TagCooccurrenceIndex.Build(images);

        var candidates = index.FindHardNegativeSources(Miku, minCooccurrenceRatio: 0.5, minCounterExamples: 15);

        var vocaloidCandidate = Assert.Single(candidates);
        Assert.Equal(Vocaloid, vocaloidCandidate.TagRow);
        Assert.Equal(1.0, vocaloidCandidate.Ratio);
        Assert.Equal(40, vocaloidCandidate.CounterExampleCount);
    }

    [Fact]
    public void FindHardNegativeSources_BreastsLargeBreastsCase_ExcludesSubsetDirection()
    {
        // large_breasts (100 images) always co-occurs with breasts; breasts itself
        // appears far more broadly (500 images total). Using large_breasts as a
        // hard-negative source FOR breasts would need images with large_breasts but
        // not breasts — there essentially are none.
        var images = new List<HashSet<int>>();
        images.AddRange(Enumerable.Repeat(0, 100).Select(_ => new HashSet<int> { Breasts, LargeBreasts }));
        images.AddRange(Enumerable.Repeat(0, 400).Select(_ => new HashSet<int> { Breasts }));

        var index = TagCooccurrenceIndex.Build(images);

        var candidates = index.FindHardNegativeSources(Breasts, minCooccurrenceRatio: 0.5, minCounterExamples: 15);

        Assert.Empty(candidates);
    }

    [Fact]
    public void FindHardNegativeSources_BreastsLargeBreastsCase_OppositeDirectionQualifies()
    {
        // Same corpus as above, but querying FOR large_breasts: breasts is a valid,
        // well-populated hard-negative source (breasts-without-large_breasts, i.e.
        // small/medium, is a large real population).
        var images = new List<HashSet<int>>();
        images.AddRange(Enumerable.Repeat(0, 100).Select(_ => new HashSet<int> { Breasts, LargeBreasts }));
        images.AddRange(Enumerable.Repeat(0, 400).Select(_ => new HashSet<int> { Breasts }));

        var index = TagCooccurrenceIndex.Build(images);

        var candidates = index.FindHardNegativeSources(LargeBreasts, minCooccurrenceRatio: 0.5, minCounterExamples: 15);

        var breastsCandidate = Assert.Single(candidates);
        Assert.Equal(Breasts, breastsCandidate.TagRow);
        Assert.Equal(1.0, breastsCandidate.Ratio);
        Assert.Equal(400, breastsCandidate.CounterExampleCount);
    }

    [Fact]
    public void FindHardNegativeSources_FiltersBelowRatioThreshold()
    {
        // Target appears in 100 images; candidate appears in 1000 images total but
        // only co-occurs with the target in 10 of them (a 10% ratio) — plenty of raw
        // counter-examples (990), but the pairing itself isn't "common" enough to trust.
        const int target = 0;
        const int candidate = 1;

        var images = new List<HashSet<int>>();
        images.AddRange(Enumerable.Repeat(0, 10).Select(_ => new HashSet<int> { target, candidate }));
        images.AddRange(Enumerable.Repeat(0, 90).Select(_ => new HashSet<int> { target }));
        images.AddRange(Enumerable.Repeat(0, 990).Select(_ => new HashSet<int> { candidate }));

        var index = TagCooccurrenceIndex.Build(images);

        var candidates = index.FindHardNegativeSources(target, minCooccurrenceRatio: 0.5, minCounterExamples: 15);

        Assert.Empty(candidates);
    }

    [Fact]
    public void FindHardNegativeSources_FiltersBelowCounterExampleThreshold()
    {
        // Target and candidate co-occur at a 100% ratio (every target image also has
        // the candidate), but the candidate is barely broader than the target — only
        // 2 counter-example images exist, below the floor of 15.
        const int target = 0;
        const int candidate = 1;

        var images = new List<HashSet<int>>();
        images.AddRange(Enumerable.Repeat(0, 50).Select(_ => new HashSet<int> { target, candidate }));
        images.AddRange(Enumerable.Repeat(0, 2).Select(_ => new HashSet<int> { candidate }));

        var index = TagCooccurrenceIndex.Build(images);

        var candidates = index.FindHardNegativeSources(target, minCooccurrenceRatio: 0.5, minCounterExamples: 15);

        Assert.Empty(candidates);
    }

    [Fact]
    public void FindHardNegativeSources_OrdersByRatioDescending()
    {
        const int target = 0;
        const int strong = 1; // co-occurs with 100% of the target's images
        const int weak = 2;   // co-occurs with 60% of the target's images

        // 100 target images total: all 100 also have `strong`, 60 of those also have
        // `weak` — both candidates measured against the SAME target total (100), not
        // against disjoint subsets, so both can legitimately clear the ratio floor at
        // once with `strong` still ranked ahead.
        var images = new List<HashSet<int>>();
        images.AddRange(Enumerable.Repeat(0, 60).Select(_ => new HashSet<int> { target, strong, weak }));
        images.AddRange(Enumerable.Repeat(0, 40).Select(_ => new HashSet<int> { target, strong }));
        images.AddRange(Enumerable.Repeat(0, 20).Select(_ => new HashSet<int> { strong })); // strong's own counter-examples
        images.AddRange(Enumerable.Repeat(0, 20).Select(_ => new HashSet<int> { weak }));   // weak's own counter-examples

        var index = TagCooccurrenceIndex.Build(images);

        var candidates = index.FindHardNegativeSources(target, minCooccurrenceRatio: 0.5, minCounterExamples: 15);

        Assert.Equal([strong, weak], candidates.Select(c => c.TagRow));
    }

    [Fact]
    public void FindHardNegativeSources_ReturnsEmpty_ForNeverObservedTargetTag()
    {
        List<HashSet<int>> images = [[1, 2]];
        var index = TagCooccurrenceIndex.Build(images);

        Assert.Empty(index.FindHardNegativeSources(targetTagRow: 999, minCooccurrenceRatio: 0, minCounterExamples: 0));
    }
}

public class NegativeQueryPlanningTests
{
    private static TagVocabulary BuildVocabulary(out int mikuRow, out int vocaloidRow)
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        mikuRow = vocabulary.RecordObservation("character:hatsune_miku").RowIndex;
        vocaloidRow = vocabulary.RecordObservation("series:vocaloid").RowIndex;
        return vocabulary;
    }

    [Fact]
    public void BuildQuerySequence_OrdersHardNegativesBeforeFallback()
    {
        var vocabulary = BuildVocabulary(out var mikuRow, out var vocaloidRow);
        var images = new List<HashSet<int>>();
        images.AddRange(Enumerable.Repeat(0, 60).Select(_ => new HashSet<int> { mikuRow, vocaloidRow }));
        images.AddRange(Enumerable.Repeat(0, 40).Select(_ => new HashSet<int> { vocaloidRow }));
        var index = TagCooccurrenceIndex.Build(images);

        var plans = NegativeQueryPlanning.BuildQuerySequence(
            "character:hatsune_miku", mikuRow, index, vocabulary,
            minCooccurrenceRatio: 0.5, minCounterExamples: 15, maxHardNegativeSources: 3);

        Assert.Equal(2, plans.Count);
        Assert.Equal("vocaloid -hatsune_miku", plans[0].TagQuery);
        Assert.Equal("negative:cooccur:series:vocaloid", plans[0].PhaseKey);
        Assert.Equal("-hatsune_miku", plans[1].TagQuery);
        Assert.Equal("negative", plans[1].PhaseKey);
    }

    [Fact]
    public void BuildQuerySequence_CapsAtMaxHardNegativeSources()
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        var targetRow = vocabulary.RecordObservation("target").RowIndex;
        var rows = Enumerable.Range(0, 5).Select(i => vocabulary.RecordObservation($"candidate{i}").RowIndex).ToList();

        // 50 shared images carry the target alongside ALL 5 candidates at once, so
        // each candidate's ratio is measured against the same 50-image target total
        // (1.0 each) rather than diluted by disjoint per-candidate image sets.
        var images = new List<HashSet<int>>();
        images.AddRange(Enumerable.Repeat(0, 50).Select(_ => new HashSet<int>(rows) { targetRow }));
        foreach (var row in rows)
            images.AddRange(Enumerable.Repeat(0, 20).Select(_ => new HashSet<int> { row })); // per-candidate counter-examples
        var index = TagCooccurrenceIndex.Build(images);

        var plans = NegativeQueryPlanning.BuildQuerySequence(
            "target", targetRow, index, vocabulary,
            minCooccurrenceRatio: 0.5, minCounterExamples: 15, maxHardNegativeSources: 2);

        // 2 hard-negative plans + 1 fallback, even though 5 candidates qualify.
        Assert.Equal(3, plans.Count);
        Assert.Equal("negative", plans[^1].PhaseKey);
    }

    [Fact]
    public void BuildQuerySequence_FallsBackToPlainQuery_WhenNoCandidatesQualify()
    {
        var vocabulary = BuildVocabulary(out var mikuRow, out _);
        var index = TagCooccurrenceIndex.Build([]);

        var plans = NegativeQueryPlanning.BuildQuerySequence(
            "character:hatsune_miku", mikuRow, index, vocabulary,
            minCooccurrenceRatio: 0.5, minCounterExamples: 15, maxHardNegativeSources: 3);

        var plan = Assert.Single(plans);
        Assert.Equal("-hatsune_miku", plan.TagQuery);
        Assert.Equal("negative", plan.PhaseKey);
    }

    [Fact]
    public void BuildQuerySequence_UnmappedTargetTag_FallsBackToPlainQueryOnly()
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        var index = TagCooccurrenceIndex.Build([[1, 2]]);

        var plans = NegativeQueryPlanning.BuildQuerySequence(
            "character:never_seen", targetTagRow: null, index, vocabulary,
            minCooccurrenceRatio: 0.5, minCounterExamples: 15, maxHardNegativeSources: 3);

        var plan = Assert.Single(plans);
        Assert.Equal("-never_seen", plan.TagQuery);
        Assert.Equal("negative", plan.PhaseKey);
    }

    [Fact]
    public void BuildQuerySequence_FallbackPhaseKeyMatchesLegacyValue()
    {
        // Guards backward compatibility with pre-existing durable TagProgress rows
        // keyed by the literal "negative" phase string.
        var vocabulary = BuildVocabulary(out var mikuRow, out _);
        var index = TagCooccurrenceIndex.Build([]);

        var plans = NegativeQueryPlanning.BuildQuerySequence(
            "character:hatsune_miku", mikuRow, index, vocabulary,
            minCooccurrenceRatio: 0.5, minCounterExamples: 15, maxHardNegativeSources: 0);

        var plan = Assert.Single(plans);
        Assert.Equal("negative", plan.PhaseKey);
    }
}
