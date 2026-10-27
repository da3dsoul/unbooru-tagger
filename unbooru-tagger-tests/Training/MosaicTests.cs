using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class MosaicTests
{
    [Fact]
    public void Compose_PutsEachImageHalfSizeInItsTile()
    {
        const int size = 8;
        // Image i is the constant i + 1, except image 2 has a 2x2 block of 9s at its top-left
        // that must average into exactly one canvas pixel of its tile.
        var pixels = new float[5 * 3 * size * size];
        for (var i = 0; i < 5; i++)
            Array.Fill(pixels, i + 1f, i * 3 * size * size, 3 * size * size);
        for (var ch = 0; ch < 3; ch++)
            foreach (var (y, x) in new[] { (0, 0), (0, 1), (1, 0), (1, 1) })
                pixels[2 * 3 * size * size + ch * size * size + y * size + x] = 9f;

        using var canvas = Mosaic.Compose(tensor(pixels, [5, 3, size, size]), [4, 2, 0, 1], size);
        Assert.Equal([1L, 3, size, size], canvas.shape);
        var values = canvas.data<float>().ToArray();
        float At(int ch, int y, int x) => values[ch * size * size + y * size + x];

        Assert.Equal(5f, At(0, 0, 0));    // tile 0 (top-left): image 4
        Assert.Equal(5f, At(2, 3, 3));
        Assert.Equal(9f, At(1, 0, 4));    // tile 1 (top-right): image 2, its 2x2 block of 9s -> one pixel
        Assert.Equal(3f, At(1, 0, 5));
        Assert.Equal(1f, At(0, 4, 0));    // tile 2 (bottom-left): image 0
        Assert.Equal(2f, At(0, 7, 7));    // tile 3 (bottom-right): image 1
    }

    [Fact]
    public void TileBox_HalvesTheContentAndMovesItIntoTheTile()
    {
        var content = new LetterboxBox(0, 64, 512, 384);
        Assert.Equal(new LetterboxBox(0, 32, 256, 192), Mosaic.TileBox(content, 0, 512));
        Assert.Equal(new LetterboxBox(256, 32, 256, 192), Mosaic.TileBox(content, 1, 512));
        Assert.Equal(new LetterboxBox(0, 288, 256, 192), Mosaic.TileBox(content, 2, 512));
        Assert.Equal(new LetterboxBox(256, 288, 256, 192), Mosaic.TileBox(content, 3, 512));
    }

    [Fact]
    public void PickTiles_UsesDistinctImages_AndOnlyWholeMosaics()
    {
        var tiles = Mosaic.PickTiles(10, 3, new Random(0));
        Assert.Equal(8, tiles.Length); // 10 images make two mosaics, not three
        Assert.Equal(8, tiles.Distinct().Count());
        Assert.Empty(Mosaic.PickTiles(3, 1, new Random(0)));
    }

    // Vocabulary: 0 = gloves, 1 = hat, 2 = a tag image 3 is likely missing, 3 = mined negative.
    private static (LocalizationPairs Pairs, PairTargets Targets) Batch()
    {
        const int vocabulary = 4;
        var labels = Enumerable.Repeat(-1f, 4 * vocabulary).ToArray();
        var weights = Enumerable.Repeat(1f, 4 * vocabulary).ToArray();
        var focal = Enumerable.Repeat(1f, 4 * vocabulary).ToArray();
        labels[0 * vocabulary + 0] = 1f;  // image 0: gloves
        labels[1 * vocabulary + 1] = 1f;  // image 1: hat
        labels[2 * vocabulary + 0] = 1f;  // image 2: gloves too
        labels[3 * vocabulary + 2] = 1f;  // image 3: tag 2...
        weights[1 * vocabulary + 2] = 0f; // ...which image 1 is likely missing
        focal[1 * vocabulary + 0] = 0f;   // image 1: gloves is a certain negative
        for (var i = 0; i < labels.Length; i++)
            if (labels[i] > 0)
                focal[i] = 0f;                // positives are always full-strength (PairWeightBuilder)

        var targets = new PairTargets(tensor(labels, [4, vocabulary]), tensor(weights, [4, vocabulary]), tensor(focal, [4, vocabulary]), 3f);
        // Selected pairs per image (width 2, weight 0 = padding): each image's positive, plus a mined negative for image 1.
        long[] rows = [0, 0, 1, 3, 0, 0, 2, 0];
        float[] pairWeights = [1, 0, 1, 1, 1, 0, 1, 0];
        var pairs = new LocalizationPairs(tensor(rows, [4, 2]), tensor(new float[8], [4, 2]), tensor(pairWeights, [4, 2]), tensor(new float[8], [4, 2]), 4f);
        return (pairs, targets);
    }

    [Fact]
    public void TilePairs_AddTheOtherTilesTagsWithThisImagesOwnLabels()
    {
        var (pairs, targets) = Batch();
        using var tiles = Mosaic.TilePairs(pairs, targets, [0, 1, 2, 3]);

        var width = (int)tiles.TagRows.shape[1];
        var rows = tiles.TagRows.data<long>().ToArray();
        var labels = tiles.Labels.data<float>().ToArray();
        var weights = tiles.Weights.data<float>().ToArray();
        var focal = tiles.FocalMask.data<float>().ToArray();
        Dictionary<long, (float Label, float Focal)> Tile(int k) => Enumerable.Range(0, width)
            .Where(j => weights[k * width + j] > 0).ToDictionary(j => rows[k * width + j], j => (labels[k * width + j], focal[k * width + j]));

        Assert.Equal(new Dictionary<long, (float, float)> { [0] = (1, 0), [1] = (-1, 1), [2] = (-1, 1) }, Tile(0));
        // Image 1: its own hat and mined negative, gloves as a certain negative, and no tag 2 (likely missing on it).
        Assert.Equal(new Dictionary<long, (float, float)> { [1] = (1, 0), [3] = (-1, 1), [0] = (-1, 0) }, Tile(1));
        // Image 2 has gloves itself: a positive in its own tile, not a negative.
        Assert.Equal((1f, 0f), Tile(2)[0]);
        Assert.Equal(4f, tiles.PositiveWeightSum); // gloves x2, hat, tag 2
    }

    [Fact]
    public void MosaicLoss_PunishesATagThatLightsUpOutsideItsOwnTile()
    {
        // One mosaic on an 8x8 map: image 0 (gloves) in tile 0, images 1-3 elsewhere without gloves.
        var (_, targets) = Batch();
        var labels = targets.Labels.data<float>().ToArray();
        labels[2 * 4 + 0] = -1f; // image 2 without gloves here
        using var noGloves = new PairTargets(tensor(labels, [4, 4]), targets.Weights, targets.FocalMask, 2f);
        using var pairs = new LocalizationPairs(tensor(new long[] { 0, 1, 3, 2 }, [4, 1]), tensor(new[] { 1f, 1f, -1f, 1f }, [4, 1]),
            tensor(new[] { 1f, 1f, 0f, 1f }, [4, 1]), tensor(new float[4], [4, 1]), 3f);
        int[] tileImages = [0, 1, 2, 3];
        using var tilePairs = Mosaic.TilePairs(pairs, noGloves, tileImages);
        var boxes = tileImages.Select((_, k) => Mosaic.TileBox(new LetterboxBox(0, 0, 64, 64), k, 64)).ToArray();
        using var mask = SpatialMask.Build(boxes, 64, 8, 8);
        float[] embeddings = [1, 0, 0, 1, 0, 1, 0, 1]; // [4 tags, 2 dims]: gloves reads dimension 0

        float Loss(Func<int, int, bool> gloves)
        {
            var map = new float[2 * 8 * 8];
            for (var y = 0; y < 8; y++)
                for (var x = 0; x < 8; x++)
                    map[y * 8 + x] = gloves(y, x) ? 4f : -4f;
            using var maps = tensor(map, [1, 2, 8, 8]).index_select(0, tensor(new long[] { 0, 0, 0, 0 }));
            using var loss = SigmoidContrastiveLoss.ComputeLocalizedPairs(maps, tensor(embeddings, [4, 2]), tilePairs, 0.25f, mask, null, 0f, 0f);
            return loss.item<float>();
        }

        var inItsTile = Loss((y, x) => y < 4 && x < 4);
        var everywhere = Loss((_, _) => true);
        var inTheWrongTile = Loss((y, x) => y >= 4 && x >= 4);
        Assert.True(inItsTile < everywhere, $"{inItsTile} vs {everywhere}");        // the other tiles' negatives
        Assert.True(inItsTile < inTheWrongTile, $"{inItsTile} vs {inTheWrongTile}");
    }
}
