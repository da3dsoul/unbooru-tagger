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
    /// batch is too small.
    /// </summary>
    public static int[] PickTiles(int batchSize, int mosaics, Random random)
    {
        var count = Math.Min(mosaics, batchSize / TilesPerMosaic) * TilesPerMosaic;
        return Enumerable.Range(0, batchSize).OrderBy(_ => random.Next()).Take(count).ToArray();
    }

    /// <summary>Where an image's content lands on the canvas as tile <paramref name="tile"/> (0-3, row-major).</summary>
    public static LetterboxBox TileBox(LetterboxBox content, int tile, int inputSize)
    {
        var half = inputSize / 2;
        var (offsetX, offsetY) = (tile % 2 * half, tile / 2 * half);
        return new LetterboxBox(offsetX + content.X / 2, offsetY + content.Y / 2, Math.Max(1, content.Width / 2), Math.Max(1, content.Height / 2));
    }

    /// <summary>[mosaics, 3, size, size] canvases from <paramref name="pixels"/> ([batch, 3, size, size]), each image 2x2 average-pooled into its tile.</summary>
    public static Tensor Compose(Tensor pixels, int[] tileImages, int inputSize)
    {
        var mosaics = tileImages.Length / TilesPerMosaic;
        var half = inputSize / 2;
        var small = nn.functional.avg_pool2d(pixels.index_select(0, tensor(tileImages.Select(i => (long)i).ToArray(), device: pixels.device)), 2);
        // [M*4, 3, h, h] -> [M, 2 rows, 2 cols, 3, h, h] -> [M, 3, 2 rows, h, 2 cols, h] -> [M, 3, size, size]
        return small.reshape([mosaics, 2, 2, 3, half, half]).permute(0, 3, 1, 4, 2, 5).reshape([mosaics, 3, inputSize, inputSize]);
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
