using UnbooruTagger.Core.Scoring;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class LabelledContextCheckTests
{
    [Fact]
    public void AbsentAndUntaggedPairs_AreCountedSeparately_AndPairsOutsideTheHeldOutImagesAreSkipped()
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        foreach (var tag in new[] { "gloves", "hat" })
            vocabulary.AddTag(tag);
        // Held-out cache rows 100, 200, 300; raw scores per image: gloves, hat.
        int[] heldOut = [100, 200, 300];
        using var logits = ImageTagLogits.Dense([2f, -1f, -3f, 0.5f, -1f, -2f], 3, new LogitScaling(1f, 0f));
        LabelledContextPair[] pairs =
        [
            new(100, "gloves", "absent"),  // called: a context false positive
            new(200, "gloves", "absent"),  // not called
            new(300, "hat", "absent"),     // not called
            new(200, "hat", "present"),    // called: an untagged hat found
            new(300, "gloves", "present"), // missed
            new(999, "gloves", "absent"),  // not a held-out image
            new(100, "scarf", "absent"),   // not in the vocabulary
        ];

        var report = LabelledContextCheck.Evaluate(pairs, heldOut, logits, vocabulary, new Dictionary<string, float>());

        Assert.Equal(3, report.Absent);
        Assert.Equal(1 / 3.0, report.AbsentCalledRate, 6);
        Assert.Equal((2 - 3 - 2) / 3.0, report.AbsentMeanLogit, 5);
        Assert.Equal(2, report.Present);
        Assert.Equal(0.5, report.PresentCalledRate, 6);
        Assert.Equal((0.5 - 1) / 2.0, report.PresentMeanLogit, 5);
        Assert.Equal(2, report.Skipped);

        // At a calibrated hat threshold of p = 0.75 (log-odds ln 3) the untagged hat (0.5) isn't called.
        var strict = LabelledContextCheck.Evaluate(pairs, heldOut, logits, vocabulary, new Dictionary<string, float> { ["hat"] = 0.75f });
        Assert.Equal(0, strict.PresentCalledRate, 6);
        Assert.Equal((0.5 - Math.Log(3) - 1) / 2.0, strict.PresentMeanLogit, 4);
    }

    [Fact]
    public void Load_RejectsAnUnknownLabel()
    {
        var path = Path.Combine(Path.GetTempPath(), $"context-check-{Guid.NewGuid():N}.json");
        try
        {
            File.WriteAllText(path, """[{"cacheIndex":1,"tag":"gloves","label":"unsure"}]""");
            Assert.Throws<InvalidDataException>(() => LabelledContextCheck.Load(path));
            File.WriteAllText(path, """[{"cacheIndex":1,"tag":"gloves","label":"present","set":"user","prior":0.7}]""");
            Assert.True(Assert.Single(LabelledContextCheck.Load(path)).Present);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
