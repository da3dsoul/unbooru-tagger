using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>How <see cref="ErasureContrast"/> trains an erased pair.</summary>
public enum ErasureLoss
{
    /// <summary>Shetty et al. 2019: the score with the object blanked must fall a margin below the score with a same-size control blanked.</summary>
    Hinge,

    /// <summary>GAIN / CSS: the object-blanked image is a certain negative for the tag, the control-blanked one (object still there) a positive.</summary>
    Negative,

    /// <summary>
    /// Only the negative half of <see cref="Negative"/>: the object-blanked image is a certain negative, the control-blanked one is
    /// trained only through the anchor (if any). <see cref="Negative"/> raised every called score, since its positive half pushes
    /// up a view the model already calls.
    /// </summary>
    NegativeOnly,
}

/// <summary>One erased pair: a batch image, a tag it carries, the canvas rectangle ([x, y, w, h]) its own map points at, and a same-size control elsewhere.</summary>
public sealed record ErasurePair(int Image, int Tag, int[] Object, int[] Control);

/// <summary>
/// Trains the classifier on the occlusion check itself. For a few localizable positives the model already calls,
/// the region the tag's own localization map points at (the live map, detached) is blanked, and so is a same-size
/// region on another of the image's tags (or, without one, at random); the two images go through the tower and the tag's score must drop more for the object than for
/// the control (<see cref="ErasureLoss"/>). Blanks are filled with a random <see cref="OcclusionFills.Training"/> fill, not a
/// flat one: with zero fills the model learned that a grey patch lowers a tag's score. A tag called from who is in the image survives the blank and is penalized;
/// one called from the object isn't. The hinge is relative, so a map that points at the wrong place gives two similar
/// views and little gradient.
/// </summary>
public static class ErasureContrast
{
    /// <summary>The erased region is capped at this fraction of the image's content, so a map spread over the character doesn't blank her.</summary>
    public const double MaxAreaFraction = 0.15;

    /// <summary>
    /// Up to <paramref name="count"/> (image, tag, other) candidates, at most one per image: a localizable positive with
    /// weight whose whole-image score is already a "yes" (scaled logit ≥ 0), so there is a call to test, and another of
    /// the image's localizable positives for the control to cover (-1 if it has none).
    /// </summary>
    /// <param name="targets">This batch's targets (CPU).</param>
    /// <param name="wholeImageLogits">[batch × vocabulary] unscaled whole-image logits, row-major.</param>
    public static List<(int Image, int Tag, int Other)> Candidates(PairTargets targets, float[] wholeImageLogits, float scale, float bias, bool[] localizable, int count, Random random)
    {
        var batch = (int)targets.Labels.shape[0];
        var vocabularySize = (int)targets.Labels.shape[1];
        var labels = targets.Labels.data<float>().ToArray();
        var weights = targets.Weights.data<float>().ToArray();
        var chosen = new List<(int, int, int)>();
        foreach (var image in Enumerable.Range(0, batch).OrderBy(_ => random.Next()))
        {
            if (chosen.Count >= count)
                break;
            var offset = image * vocabularySize;
            var positives = Enumerable.Range(0, vocabularySize).Where(t => localizable[t] && labels[offset + t] > 0 && weights[offset + t] > 0).ToList();
            var called = positives.Where(t => wholeImageLogits[offset + t] * scale + bias >= 0).ToList();
            if (called.Count == 0)
                continue;
            var tag = called[random.Next(called.Count)];
            var others = positives.Where(t => t != tag).ToList();
            chosen.Add((image, tag, others.Count > 0 ? others[random.Next(others.Count)] : -1));
        }

        return chosen;
    }

    /// <summary>
    /// The control for <paramref name="region"/>: a rectangle its size centered on the other tag's map peak (shifted inside
    /// <paramref name="content"/>), so both blanks cover the character and only which object goes differs (Shetty et al.
    /// remove another object). Null if it would overlap the region; the caller then places one at random.
    /// </summary>
    public static int[]? ControlOnOther(float[] otherHeat, bool[] valid, int grid, int cellSize, int[] content, int[] region)
    {
        var cells = Enumerable.Range(0, grid * grid).Where(i => valid[i]).ToList();
        if (cells.Count == 0 || region[2] > content[2] || region[3] > content[3])
            return null;
        var peak = cells.MaxBy(i => otherHeat[i]);
        var x = Math.Clamp(peak % grid * cellSize + cellSize / 2 - region[2] / 2, content[0], content[0] + content[2] - region[2]);
        var y = Math.Clamp(peak / grid * cellSize + cellSize / 2 - region[3] / 2, content[1], content[1] + content[3] - region[3]);
        int[] control = [x, y, region[2], region[3]];
        var overlaps = control[0] < region[0] + region[2] && region[0] < control[0] + control[2]
            && control[1] < region[1] + region[3] && region[1] < control[1] + control[3];
        return overlaps ? null : control;
    }

    /// <summary>
    /// The canvas rectangle a tag's map points at: the connected cells around its peak scoring at least halfway from the
    /// median to the peak, grown by a cell, clipped to <paramref name="content"/> and capped at <see cref="MaxAreaFraction"/>
    /// of it (a square around the peak when over). All maps are row-major [grid × grid].
    /// </summary>
    /// <param name="heat">The tag's map logits.</param>
    /// <param name="valid">Which cells are image content.</param>
    /// <param name="cellSize">Input pixels per map cell.</param>
    /// <param name="content">The content box on the canvas, [x, y, w, h].</param>
    public static int[]? Region(float[] heat, bool[] valid, int grid, int cellSize, int[] content)
    {
        var cells = Enumerable.Range(0, grid * grid).Where(i => valid[i]).ToList();
        if (cells.Count == 0)
            return null;
        var peak = cells.MaxBy(i => heat[i]);
        var sorted = cells.Select(i => heat[i]).Order().ToList();
        var threshold = heat[peak] - 0.5f * (heat[peak] - sorted[sorted.Count / 2]);

        var (minX, minY, maxX, maxY) = (peak % grid, peak / grid, peak % grid, peak / grid);
        var seen = new bool[grid * grid];
        var queue = new Queue<int>([peak]);
        seen[peak] = true;
        while (queue.TryDequeue(out var cell))
        {
            var (x, y) = (cell % grid, cell / grid);
            (minX, minY, maxX, maxY) = (Math.Min(minX, x), Math.Min(minY, y), Math.Max(maxX, x), Math.Max(maxY, y));
            foreach (var (nx, ny) in new[] { (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1) })
            {
                var next = ny * grid + nx;
                if (nx >= 0 && ny >= 0 && nx < grid && ny < grid && !seen[next] && valid[next] && heat[next] >= threshold)
                {
                    seen[next] = true;
                    queue.Enqueue(next);
                }
            }
        }

        var box = LocalizationEvaluator.Grow([minX * cellSize, minY * cellSize, (maxX - minX + 1) * cellSize, (maxY - minY + 1) * cellSize], cellSize, content);
        var limit = MaxAreaFraction * content[2] * content[3];
        if (box[2] * box[3] > limit)
        {
            var side = (int)Math.Sqrt(limit);
            var (centerX, centerY) = (peak % grid * cellSize + cellSize / 2, peak / grid * cellSize + cellSize / 2);
            box = LocalizationEvaluator.Grow([centerX - side / 2, centerY - side / 2, side, side], 0, content);
        }

        return box[2] > 0 && box[3] > 0 ? box : null;
    }

    /// <summary>
    /// [2·pairs, 3, size, size]: each pair's image with the object blanked, then with the control blanked, both
    /// filled with that pair's <paramref name="fills"/> entry (one fill per pair, so the two views differ only in where).
    /// </summary>
    public static Tensor Variants(Tensor pixels, IReadOnlyList<ErasurePair> pairs, int size, IReadOnlyList<OcclusionFill> fills)
    {
        var keep = new float[pairs.Count * 2 * size * size];
        Array.Fill(keep, 1f);
        for (var k = 0; k < pairs.Count; k++)
            foreach (var (view, r) in new[] { (2 * k, pairs[k].Object), (2 * k + 1, pairs[k].Control) })
                for (var y = Math.Max(0, r[1]); y < Math.Min(size, r[1] + r[3]); y++)
                {
                    var (start, end) = (Math.Max(0, r[0]), Math.Min(size, r[0] + r[2]));
                    if (end > start)
                        keep.AsSpan(view * size * size + y * size + start, end - start).Clear();
                }

        using var scope = NewDisposeScope();
        using var rows = tensor(pairs.SelectMany(p => new long[] { p.Image, p.Image }).ToArray(), device: pixels.device);
        var views = pixels.index_select(0, rows);
        var mask = tensor(keep, [pairs.Count * 2, 1, size, size]).to(pixels.device);
        var filled = OcclusionFills.Images(views, fills.SelectMany(f => new[] { f, f }).ToList());
        return (views * mask + filled * (1 - mask)).MoveToOuterDisposeScope();
    }

    /// <summary>
    /// The loss over the pairs' variants' pooled embeddings ([2·pairs, D], object then control per pair). With
    /// <paramref name="unblanked"/> (each pair's scaled whole-image logit, a constant) and an <paramref name="anchorWeight"/>,
    /// the control-blanked score is also held to the unblanked one (|unblanked − control|). The hinge is relative, so on its own it
    /// leaves "any filled patch lowers every score" free: from 6,000 to 12,000 steps a noise-filled control's drop grew 0.12 → 0.67.
    /// A one-sided anchor (only a drop penalized) overshot: blanking elsewhere then raised scores (−0.13) and mAP fell 2 points.
    /// </summary>
    public static Tensor Compute(Tensor pooled, Tensor tagEmbeddings, IReadOnlyList<ErasurePair> pairs, LogitScaleBias scaling, ErasureLoss loss, float margin,
        float[]? unblanked = null, float anchorWeight = 0)
    {
        using var rows = tensor(pairs.Select(p => (long)p.Tag).ToArray(), device: pooled.device);
        var embeddings = tagEmbeddings.index_select(0, rows);
        var views = pooled.reshape([pairs.Count, 2, pooled.shape[1]]);
        var objectBlanked = scaling.forward((views.select(1, 0) * embeddings).sum(1));
        var controlBlanked = scaling.forward((views.select(1, 1) * embeddings).sum(1));
        var term = loss switch
        {
            ErasureLoss.Hinge => nn.functional.relu(margin + objectBlanked - controlBlanked).mean(),
            ErasureLoss.Negative => (nn.functional.softplus(objectBlanked) + nn.functional.softplus(-controlBlanked)).mean(),
            ErasureLoss.NegativeOnly => nn.functional.softplus(objectBlanked).mean(),
            _ => throw new ArgumentOutOfRangeException(nameof(loss), loss, null),
        };
        if (unblanked is null || anchorWeight <= 0)
            return term;
        using var full = tensor(unblanked, device: pooled.device);
        return term + anchorWeight * (full - controlBlanked).abs().mean();
    }
}
