using UnbooruTagger.Training.Model;
using UnbooruTagger.Core.Encoding;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Mosaic localization training: four of a batch's images, each shrunk 2x, tiled 2x2 on one
/// canvas. Every tile is a whole image, so its tags are all still true (unlike a crop, which
/// can cut a tag out and leave the label to be learned from context), and which tile each
/// tag came from is known. The per-location loss then runs once per tile with the tile's
/// region as the bag (<see cref="SigmoidContrastiveLoss.ComputeLocalizedPairs"/>): the tile's
/// own pairs as usual, plus every other tile's positive tags as negatives here, so "gloves
/// are in that tile and not in this one" is trained even when this tile is the character who
/// usually wears them. Tiles also show every object at half size, the small-object end of
/// the scale range.
/// </summary>
public static class Mosaic
{
    public const int TilesPerMosaic = 4;

    /// <summary>
    /// Distinct batch images for up to <paramref name="mosaics"/> mosaics, four per mosaic in
    /// tile order (top-left, top-right, bottom-left, bottom-right); fewer mosaics if the
    /// batch is too small. Images drawn for a <paramref name="focus"/> anchor go first, grouped by
    /// anchor, so its positives and context negatives share a canvas: "the gloves are on that
    /// tile, not on her".
    /// </summary>
    /// <param name="sampled">Why each batch image is there, in batch order; null = pick at random.</param>
    public static int[] PickTiles(int batchSize, int mosaics, Random random, IReadOnlyList<SampledImage>? sampled = null, IReadOnlySet<int>? focus = null)
    {
        var count = Math.Min(mosaics, batchSize / TilesPerMosaic) * TilesPerMosaic;
        IEnumerable<int> order = Enumerable.Range(0, batchSize).OrderBy(_ => random.Next()).ToList();
        if (sampled is not null && focus is { Count: > 0 })
            order = order.OrderBy(i => focus.Contains(sampled[i].AnchorTag) ? 0 : 1)
                .ThenBy(i => focus.Contains(sampled[i].AnchorTag) ? sampled[i].AnchorTag : 0)
                .ThenBy(i => sampled[i].IsHardNegative ? 1 : 0);
        // Each canvas's tiles in random places, so a group's positives don't always sit top-left.
        return order.Take(count).Chunk(TilesPerMosaic).SelectMany(tiles => tiles.OrderBy(_ => random.Next())).ToArray();
    }

    /// <summary>
    /// Where an image's content lands on the canvas as tile <paramref name="tile"/> (0-3, row-major),
    /// each tile <paramref name="tileSize"/> pixels square (half the input size, or all of it).
    /// </summary>
    public static LetterboxBox TileBox(LetterboxBox content, int tile, int inputSize, int? tileSize = null)
    {
        var size = tileSize ?? inputSize / 2;
        var (offsetX, offsetY) = (tile % 2 * size, tile / 2 * size);
        return new LetterboxBox(offsetX + content.X * size / inputSize, offsetY + content.Y * size / inputSize,
            Math.Max(1, content.Width * size / inputSize), Math.Max(1, content.Height * size / inputSize));
    }

    /// <summary>
    /// [mosaics, 3, 2·tile, 2·tile] canvases from <paramref name="pixels"/> ([batch, 3, size, size]): each
    /// image 2x2 average-pooled into its tile (<paramref name="tileSize"/> = half the input, the default),
    /// or at full size on a canvas twice the input's side.
    /// </summary>
    public static Tensor Compose(Tensor pixels, int[] tileImages, int inputSize, int? tileSize = null)
    {
        var mosaics = tileImages.Length / TilesPerMosaic;
        var size = tileSize ?? inputSize / 2;
        if (size != inputSize && size * 2 != inputSize)
            throw new ArgumentException($"A tile is half the input size or all of it, not {size} of {inputSize}.", nameof(tileSize));
        var selected = pixels.index_select(0, tensor(tileImages.Select(i => (long)i).ToArray()).Upload(pixels.device));
        var tiles = size == inputSize ? selected : nn.functional.avg_pool2d(selected, 2);
        // [M*4, 3, t, t] -> [M, 2 rows, 2 cols, 3, t, t] -> [M, 3, 2 rows, t, 2 cols, t] -> [M, 3, 2t, 2t]
        return tiles.reshape([mosaics, 2, 2, 3, size, size]).permute(0, 3, 1, 4, 2, 5).reshape([mosaics, 3, 2 * size, 2 * size]);
    }

    /// <summary>
    /// The tiles' whole-image targets, for classifying each tile from the canvas's full map pooled over
    /// it: each tile image's own row of <paramref name="targets"/> (CPU). The row already labels the
    /// other tiles' tags (a context negative's "no gloves" at full strength), and the canvas's deeper
    /// layers see the neighbouring tiles, so a tile is only right if it answers from its own pixels.
    /// </summary>
    public static PairTargets TileTargets(PairTargets targets, int[] tileImages)
    {
        using var rows = tensor(tileImages.Select(i => (long)i).ToArray());
        var labels = targets.Labels.index_select(0, rows);
        var weights = targets.Weights.index_select(0, rows);
        var positiveWeightSum = (weights * labels.gt(0)).sum().item<float>();
        return new PairTargets(labels, weights, targets.FocalMask.index_select(0, rows), positiveWeightSum, targets.LogitOffsets?.index_select(0, rows));
    }

    /// <summary>
    /// Per tile: its image's own localization pairs (from <paramref name="imagePairs"/>, the
    /// batch's <see cref="LocalizationPairSelector"/> output), plus each other tile's positive
    /// tags that this image doesn't already have a pair for, labeled from this image's own
    /// targets row (usually a negative; a positive if this image carries the tag too; skipped
    /// if its weight is 0, e.g. a likely-missing label). All on the CPU, [tiles, width].
    /// </summary>
    public static LocalizationPairs TilePairs(LocalizationPairs imagePairs, PairTargets targets, int[] tileImages)
    {
        var width = (int)imagePairs.TagRows.shape[1];
        var vocabularySize = (int)targets.Labels.shape[1];
        var rows = imagePairs.TagRows.data<long>().ToArray();
        var pairWeights = imagePairs.Weights.data<float>().ToArray();
        var labels = targets.Labels.data<float>().ToArray();
        var weights = targets.Weights.data<float>().ToArray();
        var focal = targets.FocalMask.data<float>().ToArray();

        List<int> Own(int image) => Enumerable.Range(0, width).Where(k => pairWeights[image * width + k] > 0).Select(k => (int)rows[image * width + k]).ToList();
        List<int> Positives(int image) => Own(image).Where(t => labels[image * vocabularySize + t] > 0).ToList();

        var perTile = new List<int>[tileImages.Length];
        for (var k = 0; k < tileImages.Length; k++)
        {
            var image = tileImages[k];
            var tags = Own(image);
            var seen = tags.ToHashSet();
            var first = k / TilesPerMosaic * TilesPerMosaic;
            for (var other = first; other < first + TilesPerMosaic; other++)
                if (other != k)
                    foreach (var t in Positives(tileImages[other]))
                        if (weights[image * vocabularySize + t] > 0 && seen.Add(t))
                            tags.Add(t);
            perTile[k] = tags;
        }

        var tileWidth = Math.Max(1, perTile.Max(p => p.Count));
        var tagRows = new long[tileImages.Length * tileWidth];
        var tileLabels = new float[tileImages.Length * tileWidth];
        var tileWeights = new float[tileImages.Length * tileWidth];
        var tileFocal = new float[tileImages.Length * tileWidth];
        Array.Fill(tileLabels, -1f);
        var positiveWeightSum = 0f;
        for (var k = 0; k < tileImages.Length; k++)
            for (var j = 0; j < perTile[k].Count; j++)
            {
                var t = perTile[k][j];
                var source = tileImages[k] * vocabularySize + t;
                var target = k * tileWidth + j;
                tagRows[target] = t;
                tileLabels[target] = labels[source];
                tileWeights[target] = weights[source];
                tileFocal[target] = focal[source];
                if (labels[source] > 0)
                    positiveWeightSum += weights[source];
            }

        long[] shape = [tileImages.Length, tileWidth];
        return new LocalizationPairs(tensor(tagRows, shape), tensor(tileLabels, shape), tensor(tileWeights, shape), tensor(tileFocal, shape), positiveWeightSum);
    }

    /// <summary>
    /// <see cref="TilePairs"/>' pairs on the device, from device-side image pairs (<see cref="LocalizationPairSelector.SelectOnDevice"/>)
    /// whose mined negatives the CPU never sees: each tile's image's pairs gathered on the device, then the other tiles'
    /// positives (from the labels) this image doesn't already have a pair for. Duplicates of its labelled pairs are left
    /// out on the CPU; one of its mined negatives repeated as another tile's positive gets weight 0 on the device.
    /// </summary>
    /// <param name="imageLabelled">Each image's labelled pairs (positives and certain negatives).</param>
    public static LocalizationPairs TilePairsOnDevice(LocalizationPairs imagePairs, List<int>[] imageLabelled, PairTargets cpuTargets, int[] tileImages)
    {
        var device = imagePairs.TagRows.device;
        var vocabularySize = (int)cpuTargets.Labels.shape[1];
        var labels = cpuTargets.Labels.data<float>().ToArray();
        var weights = cpuTargets.Weights.data<float>().ToArray();
        var focal = cpuTargets.FocalMask.data<float>().ToArray();
        List<int> Positives(int image) => imageLabelled[image].Where(t => labels[image * vocabularySize + t] > 0).ToList();

        var extra = new List<int>[tileImages.Length];
        var positiveWeightSum = 0f;
        for (var k = 0; k < tileImages.Length; k++)
        {
            var image = tileImages[k];
            positiveWeightSum += Positives(image).Sum(t => weights[image * vocabularySize + t]);
            var seen = imageLabelled[image].ToHashSet();
            extra[k] = [];
            var first = k / TilesPerMosaic * TilesPerMosaic;
            for (var other = first; other < first + TilesPerMosaic; other++)
                if (other != k)
                    foreach (var t in Positives(tileImages[other]))
                        if (weights[image * vocabularySize + t] > 0 && seen.Add(t))
                        {
                            extra[k].Add(t);
                            if (labels[image * vocabularySize + t] > 0)
                                positiveWeightSum += weights[image * vocabularySize + t];
                        }
        }

        var extraWidth = Math.Max(1, extra.Max(p => p.Count));
        var tagRows = new long[tileImages.Length * extraWidth];
        var tileLabels = new float[tileImages.Length * extraWidth];
        var tileWeights = new float[tileImages.Length * extraWidth];
        var tileFocal = new float[tileImages.Length * extraWidth];
        Array.Fill(tileLabels, -1f);
        for (var k = 0; k < tileImages.Length; k++)
            for (var j = 0; j < extra[k].Count; j++)
            {
                var t = extra[k][j];
                var source = tileImages[k] * vocabularySize + t;
                var target = k * extraWidth + j;
                tagRows[target] = t;
                tileLabels[target] = labels[source];
                tileWeights[target] = weights[source];
                tileFocal[target] = focal[source];
            }

        using var scope = NewDisposeScope();
        long[] shape = [tileImages.Length, extraWidth];
        var images = tensor(tileImages.Select(i => (long)i).ToArray()).Upload(device);
        var ownRows = imagePairs.TagRows.index_select(0, images);
        var ownWeights = imagePairs.Weights.index_select(0, images);
        var extraRows = tensor(tagRows, shape).Upload(device);
        var repeated = extraRows.unsqueeze(2).eq(ownRows.unsqueeze(1)).logical_and(ownWeights.gt(0).unsqueeze(1)).any(2);
        var extraWeights = tensor(tileWeights, shape).Upload(device) * repeated.logical_not().to_type(ScalarType.Float32);
        var pairs = new LocalizationPairs(
            cat([ownRows, extraRows], 1),
            cat([imagePairs.Labels.index_select(0, images), tensor(tileLabels, shape).Upload(device)], 1),
            cat([ownWeights, extraWeights], 1),
            cat([imagePairs.FocalMask.index_select(0, images), tensor(tileFocal, shape).Upload(device)], 1),
            positiveWeightSum);
        scope.MoveToOuter(pairs.TagRows, pairs.Labels, pairs.Weights, pairs.FocalMask);
        return pairs;
    }

    /// <summary>
    /// Mosaic glimpse labels (<c>--mosaic-glimpse-weight</c>), on the CPU: each canvas's localizable tags that some tile carries
    /// (a positive with weight), at most <paramref name="tagsPerCanvas"/> (a random subset beyond that; padded to exactly that many,
    /// so the term's shapes don't change from step to step), and per tile what its image says about each: 1 a positive, 0 a
    /// labelled "no" (weight above 0), -1 unknown (weight 0: likely missing, site-blind).
    /// </summary>
    /// <returns>Rows [canvases·tags], which are real [canvases·tags], tile labels [canvases·4·tags].</returns>
    public static (long[] Rows, bool[] Valid, float[] TileLabels) GlimpseTileLabels(PairTargets targets, int[] tileImages, IReadOnlyList<bool> localizable, int tagsPerCanvas, Random random)
    {
        var vocabularySize = (int)targets.Labels.shape[1];
        var labels = targets.Labels.data<float>().ToArray();
        var weights = targets.Weights.data<float>().ToArray();
        var canvases = tileImages.Length / TilesPerMosaic;
        var rows = new long[canvases * tagsPerCanvas];
        var valid = new bool[canvases * tagsPerCanvas];
        var tileLabels = new float[canvases * TilesPerMosaic * tagsPerCanvas];
        Array.Fill(tileLabels, -1f);
        for (var c = 0; c < canvases; c++)
        {
            var tiles = tileImages.Skip(c * TilesPerMosaic).Take(TilesPerMosaic).ToArray();
            var tags = tiles.SelectMany(image => Enumerable.Range(0, vocabularySize)
                    .Where(t => localizable[t] && labels[image * vocabularySize + t] > 0 && weights[image * vocabularySize + t] > 0))
                .Distinct().OrderBy(_ => random.Next()).Take(tagsPerCanvas).ToArray();
            for (var j = 0; j < tags.Length; j++)
            {
                rows[c * tagsPerCanvas + j] = tags[j];
                valid[c * tagsPerCanvas + j] = true;
                for (var q = 0; q < TilesPerMosaic; q++)
                {
                    var source = tiles[q] * vocabularySize + tags[j];
                    tileLabels[(c * TilesPerMosaic + q) * tagsPerCanvas + j] = weights[source] <= 0 ? -1f : labels[source] > 0 ? 1f : 0f;
                }
            }
        }
        return (rows, valid, tileLabels);
    }

    /// <summary>
    /// [C, K, 4] each window's overlap with each tile (any overlap) and [C, K, 4] whether it lies wholly inside it, on the device,
    /// for windows <paramref name="choice"/> on canvases of 2x2 tiles of <paramref name="tileSize"/> pixels.
    /// </summary>
    public static (Tensor Overlaps, Tensor Inside) WindowTiles(GlimpseChoice choice, int tileSize)
    {
        using var scope = NewDisposeScope();
        var device = choice.X.device;
        var side = tensor(Enumerable.Range(0, Glimpses.Zooms.Length).Select(z => (long)Glimpses.SideOf(z)).ToArray()).to(device).index_select(0, choice.Zoom.flatten()).view_as(choice.Zoom);
        var (x0, y0) = (choice.X.unsqueeze(2), choice.Y.unsqueeze(2));
        var (x1, y1) = (x0 + side.unsqueeze(2), y0 + side.unsqueeze(2));
        var tileX = tensor(new long[] { 0, tileSize, 0, tileSize }).to(device).view(1, 1, 4);
        var tileY = tensor(new long[] { 0, 0, tileSize, tileSize }).to(device).view(1, 1, 4);
        var overlaps = x0.lt(tileX + tileSize).logical_and(x1.gt(tileX)).logical_and(y0.lt(tileY + tileSize)).logical_and(y1.gt(tileY));
        var inside = x0.ge(tileX).logical_and(x1.le(tileX + tileSize)).logical_and(y0.ge(tileY)).logical_and(y1.le(tileY + tileSize));
        var valid = choice.Valid.unsqueeze(2);
        return (overlaps.logical_and(valid).MoveToOuterDisposeScope(), inside.logical_and(valid).MoveToOuterDisposeScope());
    }

    /// <summary>
    /// The mosaic glimpse term: on each canvas, a tag carried by one tile is a "no" in every window that touches only tiles
    /// whose image says no (per window, not pooled: the window-level supervision the gold-box ceiling test showed the sensor
    /// can use, 2026-11-02), and a "yes" for the log-sum-exp (<paramref name="temperature"/>, as the classifier pools) over the
    /// windows wholly inside a tile that carries it. Mean "no" BCE plus mean "yes" BCE.
    /// </summary>
    /// <param name="cells">[C, D, K] the canvases' glimpse embeddings.</param>
    /// <param name="tags">[C, T, D] each canvas's tag rows (<see cref="GlimpseTileLabels"/>).</param>
    /// <param name="tagValid">[C, T].</param>
    /// <param name="tileLabels">[C, 4, T].</param>
    public static Tensor GlimpseTerm(Tensor cells, Tensor tags, Tensor tagValid, Tensor tileLabels, Tensor overlaps, Tensor inside, float temperature, LogitScaleBias scaling)
    {
        using var scope = NewDisposeScope();
        var dots = tags.bmm(cells.to_type(tags.dtype)); // [C, T, K]
        // [C, T, K, 4] -> a window is a "no" when every tile it touches says no, a bag member when wholly inside a "yes" tile.
        var touches = overlaps.unsqueeze(1); // [C, 1, K, 4]
        var labels = tileLabels.transpose(1, 2).unsqueeze(2); // [C, T, 1, 4]
        var negative = touches.logical_and(labels.ne(0f)).any(3).logical_not().logical_and(overlaps.any(2).unsqueeze(1)).logical_and(tagValid.unsqueeze(2));
        var member = inside.unsqueeze(1).logical_and(labels.eq(1f)).any(3).logical_and(tagValid.unsqueeze(2)); // [C, T, K]

        var logits = scaling.forward(dots);
        var negativeLoss = (nn.functional.binary_cross_entropy_with_logits(logits, zeros_like(logits), reduction: nn.Reduction.None) * negative).sum()
                           / negative.sum().clamp_min(1);
        var count = member.sum(2); // [C, T]
        // Finite, not -inf: a tag without a bag would otherwise send NaN through the where() below.
        var masked = where(member, dots / temperature, full_like(dots, -1e4f));
        var hasBag = count.gt(0);
        var pooled = temperature * (masked.logsumexp(2) - count.clamp_min(1).to_type(dots.dtype).log()); // [C, T]
        var bagLogits = scaling.forward(where(hasBag, pooled, zeros_like(pooled)));
        var positiveLoss = (nn.functional.binary_cross_entropy_with_logits(bagLogits, ones_like(bagLogits), reduction: nn.Reduction.None) * hasBag).sum()
                           / hasBag.sum().clamp_min(1);
        return (negativeLoss + positiveLoss).MoveToOuterDisposeScope();
    }
}
