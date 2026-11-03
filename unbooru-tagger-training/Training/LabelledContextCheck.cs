using System.Text.Json;
using System.Text.Json.Serialization;
using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Training.Training;

/// <summary>One hand-labelled context negative: an image not tagged with <paramref name="Tag"/> whose character/artist/series usually has it, checked by eye.</summary>
/// <param name="Present">The object is visible (a missing label), not absent.</param>
/// <param name="Set">Who labelled it ("user", "agent"), so a set can be read on its own.</param>
public sealed record LabelledContextPair(
    [property: JsonPropertyName("cacheIndex")] int CacheIndex,
    [property: JsonPropertyName("tag")] string Tag,
    [property: JsonPropertyName("label")] string Label,
    [property: JsonPropertyName("set")] string? Set = null)
{
    [JsonIgnore]
    public bool Present => Label == "present";
}

/// <summary>Whole-image calls on hand-labelled context negatives (<see cref="LabelledContextCheck"/>).</summary>
/// <param name="Absent">Pairs scored where the object is absent.</param>
/// <param name="AbsentCalledRate">Of those, the fraction called at the tag's calibrated threshold: true context false positives.</param>
/// <param name="AbsentMeanLogit">Their mean log-odds minus the threshold's: below 0 = not called on average, lower = surer the object isn't there.</param>
/// <param name="Present">Pairs scored where the object is visible but untagged.</param>
/// <param name="PresentCalledRate">Of those, the fraction called: untagged objects found.</param>
/// <param name="Skipped">Labelled pairs not scored: the image isn't in this evaluation's held-out images, or the tag isn't in the vocabulary.</param>
public sealed record LabelledContextReport(
    int Absent, double AbsentCalledRate, double AbsentMeanLogit,
    int Present, double PresentCalledRate, double PresentMeanLogit,
    int Skipped);

/// <summary>
/// Context false positives measured on images checked by eye. The context-reliance bands (<see cref="ContextRelianceEvaluator"/>)
/// count every untagged image of a character who usually has a tag as a negative, but many of those show the object without the tag:
/// of 180 high-prior negatives checked by hand, 27 showed it, 50 couldn't be told and 103 didn't. As a model gets better at finding
/// untagged objects, the bands' false-positive rate rises (run 4: 20% → 25% from 170K to 1,030K steps) while calls where the object
/// is really absent fall. This scores the labelled pairs whose images are among the held-out images being evaluated, called at
/// each tag's calibrated threshold, as the band check and a deployed model call them.
/// </summary>
public static class LabelledContextCheck
{
    public static IReadOnlyList<LabelledContextPair> Load(string path)
    {
        var pairs = JsonSerializer.Deserialize<List<LabelledContextPair>>(File.ReadAllText(path))
            ?? throw new InvalidDataException($"'{path}' isn't a list of labelled context pairs.");
        if (pairs.FirstOrDefault(p => p.Label is not ("present" or "absent")) is { } bad)
            throw new InvalidDataException($"'{path}': label '{bad.Label}' for {bad.Tag} on image {bad.CacheIndex} isn't 'present' or 'absent'.");
        return pairs;
    }

    /// <param name="heldOut">The evaluated images' cache rows, in <paramref name="logits"/>' order.</param>
    /// <param name="thresholds">Each tag's calibrated probability threshold (<see cref="TagEvaluation.Threshold"/>); 0.5 for a tag without one.</param>
    public static LabelledContextReport Evaluate(
        IReadOnlyList<LabelledContextPair> pairs, IReadOnlyList<int> heldOut, ImageTagLogits logits, TagVocabulary vocabulary,
        IReadOnlyDictionary<string, float> thresholds)
    {
        var position = new Dictionary<int, int>(heldOut.Count);
        for (var i = 0; i < heldOut.Count; i++)
            position[heldOut[i]] = i;

        var absent = new List<float>();
        var present = new List<float>();
        var skipped = 0;
        foreach (var pair in pairs)
        {
            if (!position.TryGetValue(pair.CacheIndex, out var image) || !vocabulary.TryGet(pair.Tag, out var record))
            {
                skipped++;
                continue;
            }
            var threshold = thresholds.TryGetValue(pair.Tag, out var t) ? Math.Clamp(t, 1e-6f, 1 - 1e-6f) : 0.5f;
            (pair.Present ? present : absent).Add(logits.Logit(image, record.RowIndex) - MathF.Log(threshold / (1 - threshold)));
        }

        return new LabelledContextReport(
            absent.Count, CalledRate(absent), Mean(absent),
            present.Count, CalledRate(present), Mean(present),
            skipped);
    }

    private static double CalledRate(List<float> logits) => logits.Count > 0 ? logits.Count(l => l >= 0) / (double)logits.Count : double.NaN;

    private static double Mean(List<float> logits) => logits.Count > 0 ? logits.Average() : double.NaN;
}
