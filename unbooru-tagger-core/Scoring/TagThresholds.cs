using System.Text.Json;

namespace UnbooruTagger.Core.Scoring;

/// <summary>
/// Per-tag confidence cutoffs calibrated on held-out data by the training CLI's
/// <c>evaluate</c> command (each tag's best-F1 threshold against its own positives and
/// hard negatives). One global cutoff can't fit 25K tags with very different base rates
/// and difficulty: it's simultaneously too loose for tags the model over-fires on and too
/// strict for tags it under-scores. Keyed by tag name rather than row index so a file
/// stays readable and survives being hand-edited.
/// </summary>
public sealed class TagThresholds
{
    public const string FileName = "tag_thresholds.json";

    private readonly Dictionary<string, float> _byTag;

    public TagThresholds(IReadOnlyDictionary<string, float> byTag) => _byTag = new Dictionary<string, float>(byTag);

    public IReadOnlyDictionary<string, float> ByTag => _byTag;

    /// <summary>The calibrated cutoff for <paramref name="tag"/>, or <paramref name="fallback"/> if it was never calibrated (too few held-out positives).</summary>
    public float For(string tag, float fallback) => _byTag.GetValueOrDefault(tag, fallback);

    /// <summary>Null if <paramref name="path"/> doesn't exist — a bundle exported before calibration existed keeps using one global threshold.</summary>
    public static TagThresholds? Load(string path) =>
        File.Exists(path)
            ? new TagThresholds(JsonSerializer.Deserialize<Dictionary<string, float>>(File.ReadAllText(path)) ?? [])
            : null;

    public void Save(string path) =>
        File.WriteAllText(path, JsonSerializer.Serialize(_byTag.OrderBy(kv => kv.Key, StringComparer.Ordinal).ToDictionary(), new JsonSerializerOptions { WriteIndented = true }));
}
