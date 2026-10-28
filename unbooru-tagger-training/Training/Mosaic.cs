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
        var selected = pixels.index_select(0, tensor(tileImages.Select(i => (long)i).ToArray(), device: pixels.device));
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
}
