using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>One image a character-extent term trains: its batch row, its one character, and its other labelled localizable tags.</summary>
public sealed record CharacterExtentImage(int Image, int Character, int[] Others);

/// <summary>
/// Two label-free ways to make a character's localization map cover the character, not only the face (2026-11-04, the user's
/// idea: a character is a contiguous entity). The map's MIL loss pools at a low temperature, so it rewards a tag's single most
/// telling spot, and for a character that is the face: cut 4 log-odds below its peak, a character's map covered 4% of its gold
/// box (region precision 0.86), on every model from run 1 to run 5. Both terms train only images labelled solo with one character,
/// where the figure is that character's.
/// <list type="bullet">
/// <item><see cref="ExtentTerm"/> (<c>--character-extent-weight</c>): the figure is the cells within <see cref="FigureBand"/>
/// log-odds of the peak of one of the image's other labelled localizable tags (hair, clothes, accessories; the live map,
/// detached). The character's map is pushed so figure cells sit within <see cref="ExtentBand"/> of its own (detached) peak and
/// cells outside it below that band, a ring of <see cref="IgnoreRing"/> cells around the figure left out. Relative to the peak,
/// with the logit scale held, so it shapes the map without moving its calibration.</item>
/// <item><see cref="HideTerm"/> (<c>--character-hide-weight</c>, hide-and-seek, Singh &amp; Lee 2017; adversarial erasing, Wei
/// et al. 2017): the region the character's map points at is blanked (<see cref="ErasureContrast.Region"/>), the view goes
/// through the tower, and the character is a positive bag on its localization map, so the rest of the character has to carry it.</item>
/// </list>
/// Neither is used (A/B, 12,000 steps from run 4 at 2,257,658): the extent term at 0.05 flattened every tag's map (region recall 0.99,
/// pointing 6%, mAP 0.80 -> 0.16; it starts at 10-20 per image and its gradient runs through the shared map features), and
/// hide-and-seek changed nothing (mAP and region recall equal to the control). Boxes come from the figure instead (TagScorer.FigureBox).
/// </summary>
public static class CharacterExtent
{
    public const float FigureBand = 3f;
    public const float ExtentBand = 4f;
    public const int IgnoreRing = 2;
    /// <summary>Other labelled tags per image the figure is cut from (rows padded to this width).</summary>
    public const int MaxOthers = 16;

    /// <summary>
    /// Images labelled <paramref name="soloRow"/> with exactly one positive character, at most <paramref name="max"/> (0 = all),
    /// with their other localizable positives (at most <see cref="MaxOthers"/>). <paramref name="needOthers"/>: skip images without any.
    /// </summary>
    public static List<CharacterExtentImage> Select(PairTargets targets, bool[] localizable, bool[] isCharacter, int soloRow, int max, bool needOthers, Random random)
    {
        var batch = (int)targets.Labels.shape[0];
        var vocabularySize = (int)targets.Labels.shape[1];
        var labels = targets.Labels.data<float>().ToArray();
        var weights = targets.Weights.data<float>().ToArray();
        var chosen = new List<CharacterExtentImage>();
        if (soloRow < 0)
            return chosen;
        foreach (var image in Enumerable.Range(0, batch).OrderBy(_ => random.Next()))
        {
            if (max > 0 && chosen.Count >= max)
                break;
            var offset = image * vocabularySize;
            if (labels[offset + soloRow] <= 0)
                continue;
            var positives = Enumerable.Range(0, vocabularySize).Where(t => labels[offset + t] > 0 && weights[offset + t] > 0).ToList();
            var characters = positives.Where(t => isCharacter[t]).ToList();
            if (characters.Count != 1)
                continue;
            var others = positives.Where(t => localizable[t] && !isCharacter[t]).OrderBy(_ => random.Next()).Take(MaxOthers).ToArray();
            if (needOthers && others.Length == 0)
                continue;
            chosen.Add(new(image, characters[0], others));
        }

        return chosen;
    }

    /// <summary>The scaled-logit units of one map dot product, without a gradient into the shared scale.</summary>
    private static Tensor Scale(LogitScaleBias scaling, Device device)
    {
        using var two = tensor(new[] { 0f, 1f }, device: device);
        var both = scaling.forward(two);
        return (both[1] - both[0]).detach();
    }

    /// <summary>
    /// The extent term over <paramref name="images"/> (each must have others). <paramref name="localizationMap"/> is the live
    /// [batch, D, g, g] map, <paramref name="spatialMask"/> [batch, 1, g, g] its content cells.
    /// </summary>
    public static Tensor ExtentTerm(Tensor localizationMap, Tensor spatialMask, Tensor tagEmbeddings, LogitScaleBias scaling, IReadOnlyList<CharacterExtentImage> images)
    {
        var device = localizationMap.device;
        var (dim, grid) = (localizationMap.shape[1], localizationMap.shape[2]);
        var cells = grid * localizationMap.shape[3];
        var k = images.Count;
        using var rows = tensor(images.Select(i => (long)i.Image).ToArray(), device: device);
        using var characterRows = tensor(images.Select(i => (long)i.Character).ToArray(), device: device);
        var otherRows = new long[k * MaxOthers];
        var otherValid = new float[k * MaxOthers];
        for (var i = 0; i < k; i++)
            for (var j = 0; j < images[i].Others.Length; j++)
            {
                otherRows[i * MaxOthers + j] = images[i].Others[j];
                otherValid[i * MaxOthers + j] = 1;
            }
        using var others = tensor(otherRows, device: device);
        using var othersValid = tensor(otherValid, [k, MaxOthers, 1], device: device);

        var maps = localizationMap.index_select(0, rows).reshape([k, dim, cells]);
        var content = spatialMask.index_select(0, rows).reshape([k, 1, cells]).gt(0);
        var scale = Scale(scaling, device);
        var characterDot = tagEmbeddings.index_select(0, characterRows).unsqueeze(1).bmm(maps.to_type(tagEmbeddings.dtype)); // [k, 1, cells]

        Tensor figure, peak;
        using (no_grad())
        {
            var otherDot = tagEmbeddings.index_select(0, others).reshape([k, MaxOthers, dim]).bmm(maps.detach().to_type(tagEmbeddings.dtype)); // [k, K, cells]
            var masked = otherDot.masked_fill(content.logical_not(), float.NegativeInfinity);
            var otherPeak = masked.amax([2], keepdim: true);
            var inFigure = ((otherDot - otherPeak) * scale).ge(-FigureBand).logical_and(othersValid.gt(0)).any(1, keepdim: true).logical_and(content); // [k, 1, cells]
            figure = inFigure.to_type(ScalarType.Float32);
            peak = characterDot.detach().masked_fill(content.logical_not(), float.NegativeInfinity).amax([2], keepdim: true);
        }

        using var ring = nn.functional.max_pool2d(figure.reshape([k, 1, grid, cells / grid]), 2 * IgnoreRing + 1, 1, IgnoreRing).reshape([k, 1, cells]);
        using var outside = content.to_type(ScalarType.Float32) * (1 - ring);
        // Past the band below the peak, in scaled log-odds: positive inside the band.
        var relative = (characterDot - peak) * scale + ExtentBand;
        var inside = (nn.functional.softplus(-relative) * figure).sum(2) / figure.sum(2).clamp_min(1);
        var beyond = (nn.functional.softplus(relative) * outside).sum(2) / outside.sum(2).clamp_min(1);
        return (inside + beyond).mean();
    }

    /// <summary>
    /// [images, 3, size, size]: each image with <paramref name="regions"/>' rectangle ([x, y, w, h], canvas pixels) filled with its
    /// <paramref name="fills"/> entry.
    /// </summary>
    public static Tensor Hidden(Tensor pixels, IReadOnlyList<CharacterExtentImage> images, IReadOnlyList<int[]> regions, int size, IReadOnlyList<OcclusionFill> fills)
    {
        var keep = new float[images.Count * size * size];
        Array.Fill(keep, 1f);
        for (var k = 0; k < images.Count; k++)
        {
            var r = regions[k];
            for (var y = Math.Max(0, r[1]); y < Math.Min(size, r[1] + r[3]); y++)
            {
                var (start, end) = (Math.Max(0, r[0]), Math.Min(size, r[0] + r[2]));
                if (end > start)
                    keep.AsSpan(k * size * size + y * size + start, end - start).Clear();
            }
        }

        using var scope = NewDisposeScope();
        using var rows = tensor(images.Select(i => (long)i.Image).ToArray(), device: pixels.device);
        var views = pixels.index_select(0, rows);
        var mask = tensor(keep, [images.Count, 1, size, size]).to(pixels.device);
        return (views * mask + OcclusionFills.Images(views, fills) * (1 - mask)).MoveToOuterDisposeScope();
    }

    /// <summary>
    /// The hide-and-seek term: each hidden view's character as a positive bag on the view's localization map, pooled as the MIL loss
    /// pools (log-sum-exp at <paramref name="temperature"/>, in dot units, over the content cells).
    /// </summary>
    public static Tensor HideTerm(Tensor hiddenMap, Tensor hiddenMask, Tensor tagEmbeddings, LogitScaleBias scaling, IReadOnlyList<CharacterExtentImage> images, float temperature)
    {
        var device = hiddenMap.device;
        var (k, dim) = (hiddenMap.shape[0], hiddenMap.shape[1]);
        var cells = hiddenMap.shape[2] * hiddenMap.shape[3];
        using var characterRows = tensor(images.Select(i => (long)i.Character).ToArray(), device: device);
        var dot = tagEmbeddings.index_select(0, characterRows).unsqueeze(1).bmm(hiddenMap.reshape([k, dim, cells]).to_type(tagEmbeddings.dtype)).squeeze(1); // [k, cells]
        var mask = hiddenMask.reshape([k, cells]);
        var scaled = (dot / temperature).masked_fill(mask.eq(0), float.NegativeInfinity);
        var pooled = temperature * (scaled.logsumexp(1) - mask.sum(1).log());
        return nn.functional.softplus(-scaling.forward(pooled)).mean();
    }
}
