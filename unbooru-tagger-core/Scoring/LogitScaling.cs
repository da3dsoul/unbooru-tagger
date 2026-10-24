using System.Text.Json;

namespace UnbooruTagger.Core.Scoring;

/// <summary>
/// The scale and bias applied to a raw image-tag dot product before <see cref="TagScorer"/>'s
/// sigmoid — <c>sigmoid(Scale * dot + Bias)</c> — mirroring the SigLIP paper's own
/// parametrization. Needed because almost every (image, tag) pair in a batch is a
/// negative: with a vocabulary in the tens of thousands and small batches, an
/// unscaled/unbiased sigmoid has no way to represent "predict negative by default"
/// except by shrinking a rare tag's own embedding, which starves the long tail of
/// positive gradient (see CLAUDE.md's training-objective section). Learned during
/// training (<c>UnbooruTagger.Training.Model.LogitScaleBias</c>) and exported alongside
/// the rest of <see cref="Runtime.ModelBundle"/> so inference reproduces the exact
/// operating point training converged to.
/// </summary>
public sealed record LogitScaling(float Scale, float Bias)
{
    /// <summary>Scale=1, Bias=0 — the plain <c>sigmoid(dot)</c> behavior every model bundle exported before this type existed implicitly had.</summary>
    public static readonly LogitScaling Identity = new(1f, 0f);

    public float Apply(float logit) => Scale * logit + Bias;

    /// <summary>Falls back to <see cref="Identity"/> if <paramref name="path"/> doesn't exist, so an older exported model bundle still loads and scores exactly as it always did.</summary>
    public static LogitScaling Load(string path) =>
        File.Exists(path) ? JsonSerializer.Deserialize<LogitScaling>(File.ReadAllText(path)) ?? Identity : Identity;

    public void Save(string path) => File.WriteAllText(path, JsonSerializer.Serialize(this));
}
