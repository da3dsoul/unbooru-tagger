namespace UnbooruTagger.Core.Scoring;

/// <summary>
/// A rough region (pixel coordinates in the original, un-resized image) where a tag's
/// evidence concentrates, plus the peak heatmap confidence within that region. This is
/// derived from the same MaskCLIP-style spatial dot product as <see cref="TagScorer.Heatmap"/>
/// — see CLAUDE.md's localization section — not a trained detector, so treat it as
/// approximate, not a tight box.
/// </summary>
public sealed record BoundingBox(int X, int Y, int Width, int Height, float Confidence);

/// <summary>One tagged concept found in an image: its whole-image confidence plus every region it was localized to.</summary>
public sealed record TagDetection(string Tag, float Confidence, IReadOnlyList<BoundingBox> Boxes);
