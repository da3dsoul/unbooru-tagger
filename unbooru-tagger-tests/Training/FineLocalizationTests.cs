using UnbooruTagger.Core.Embedding;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Checkpoints;
using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class FineLocalizationTests
{
    private const int EmbeddingDim = 8;

    private static ImageTower Tower() =>
        new(EmbeddingDim, stemChannels: 8, stageChannels: [8, 12, 16], blocksPerStage: [1, 1, 1], fusedStages: 2) { LocalizationFromFusedOnly = true };

    private static Tensor Upsample2x(Tensor map) =>
        nn.functional.interpolate(map, size: [map.shape[2] * 2, map.shape[3] * 2], mode: InterpolationMode.Nearest);

    /// <summary>A content mask on the coarse grid (letterboxed: the right quarter is padding).</summary>
    private static Tensor CoarseMask(long batch, long side)
    {
        var mask = ones(batch, 1, side, side);
        mask[.., .., .., TensorIndex.Slice(side * 3 / 4)] = 0;
        return mask;
    }

    [Fact]
    public void ANewLevel_LeavesScoresUnchanged_AndHalvesTheStride()
    {
        var tower = Tower();
        tower.eval();
        var input = rand(2, 3, 64, 64);
        var coarseMask = CoarseMask(2, 64 / tower.OutputStride);
        var (pooled, localization) = tower.ForwardPooled(input, coarseMask);
        Assert.Equal(4, tower.OutputStride);

        tower.Fine = tower.NewFineLocalization();
        Assert.Equal(2, tower.OutputStride);
        var fineMask = Upsample2x(coarseMask);
        var (finePooled, fineLocalization) = tower.ForwardPooled(input, fineMask);

        Assert.Equal(64 / tower.OutputStride, fineLocalization.shape[2]);
        Assert.True(finePooled.allclose(pooled, rtol: 1e-4, atol: 1e-5));
        Assert.True(fineLocalization.allclose(Upsample2x(localization), rtol: 1e-4, atol: 1e-5));
    }

    [Fact]
    public void ForwardPooled_MatchesForwardMaps_WithATrainedLevel()
    {
        var tower = Tower();
        tower.eval();
        tower.Fine = tower.NewFineLocalization();
        using (no_grad())
        {
            tower.Fine.Projection.weight!.normal_(0, 0.1);
            tower.Fine.Block.LayerScale.fill_(0.5f);
        }
        var input = rand(2, 3, 64, 64);
        var mask = Upsample2x(CoarseMask(2, 64 / 4));

        var (pooled, localization) = tower.ForwardPooled(input, mask);
        var (spatial, mapsLocalization) = tower.ForwardMaps(input);

        Assert.Equal(spatial.shape, mapsLocalization.shape);
        Assert.True(localization.allclose(mapsLocalization, rtol: 1e-4, atol: 1e-5));
        Assert.True(pooled.allclose(ImageTower.MaskedPool(spatial, mask), rtol: 1e-4, atol: 1e-5));
        Assert.True(pooled.allclose(tower.ForwardPooledOnly(input, mask), rtol: 1e-4, atol: 1e-5));
        // The level changes the localization map, not classification.
        tower.Fine = null;
        var (coarsePooled, coarseLocalization) = tower.ForwardPooled(input, CoarseMask(2, 64 / 4));
        Assert.True(pooled.allclose(coarsePooled, rtol: 1e-4, atol: 1e-5));
        Assert.False(localization.allclose(Upsample2x(coarseLocalization), rtol: 1e-4, atol: 1e-5));
    }

    /// <summary>Random parts of a fine map (coarse 4x4, fine 8x8), with gradients, and pairs over a 6-tag table.</summary>
    private static (FineLocalizationMap Map, Tensor[] Leaves, Tensor Tags, LocalizationPairs Pairs) RandomFineMap(long batch)
    {
        manual_seed(7);
        var coarse = randn(batch, EmbeddingDim, 4, 4).requires_grad_();
        var features = randn(batch, 5, 8, 8).requires_grad_();
        var weight = (randn(EmbeddingDim, 5) * 0.5f).requires_grad_();
        var bias = (randn(EmbeddingDim) * 0.1f).requires_grad_();
        var tags = randn(6, EmbeddingDim).requires_grad_();
        var rows = new long[batch * 3];
        var labels = new float[batch * 3];
        for (var i = 0; i < rows.Length; i++)
        {
            rows[i] = (i * 5 + 1) % 6;
            labels[i] = i % 3 == 0 ? 1f : -1f;
        }
        var pairs = new LocalizationPairs(tensor(rows, [batch, 3]), tensor(labels, [batch, 3]), ones(batch, 3), zeros(batch, 3), batch);
        return (new FineLocalizationMap(coarse, features, weight, bias), [coarse, features, weight, bias, tags], tags, pairs);
    }

    private static Tensor[] Gradients(Tensor loss, Tensor[] leaves)
    {
        foreach (var leaf in leaves)
            leaf.grad?.zero_();
        loss.backward();
        return leaves.Select(l => l.grad!.clone()).ToArray();
    }

    [Fact]
    public void TheLossFromTheParts_MatchesTheDenseMap()
    {
        var (map, leaves, tags, pairs) = RandomFineMap(2);
        // Letterboxed at the fine grid: image 0 loses its right three fine columns (one whole coarse column and half of
        // another), image 1 its bottom row.
        var mask = ones(2, 1, 8, 8);
        mask[0, .., .., TensorIndex.Slice(5)] = 0;
        mask[1, .., TensorIndex.Slice(7), ..] = 0;
        var scaling = new LogitScaleBias();

        var dense = SigmoidContrastiveLoss.ComputeLocalizedPairs(map.Dense(), tags, pairs, 0.25f, mask, scaling, 2f, 0.05f);
        var denseGradients = Gradients(dense, leaves);
        var factored = SigmoidContrastiveLoss.ComputeLocalizedPairs(map, tags, pairs, 0.25f, mask, scaling, 2f, 0.05f);
        var factoredGradients = Gradients(factored, leaves);

        Assert.Equal(dense.item<float>(), factored.item<float>(), 4);
        for (var i = 0; i < leaves.Length; i++)
            Assert.True(factoredGradients[i].allclose(denseGradients[i], rtol: 1e-3, atol: 1e-5), $"gradient {i}");
    }

    [Fact]
    public void TileCrops_GiveTheWholeCanvasLoss()
    {
        var (canvas, leaves, tags, _) = RandomFineMap(1);
        // Four tiles on one canvas, each letterboxed inside its own quadrant.
        var tileMask = zeros(4, 1, 8, 8);
        tileMask[0, .., TensorIndex.Slice(0, 4), TensorIndex.Slice(0, 3)] = 1;
        tileMask[1, .., TensorIndex.Slice(1, 4), TensorIndex.Slice(4, 8)] = 1;
        tileMask[2, .., TensorIndex.Slice(4, 8), TensorIndex.Slice(0, 4)] = 1;
        tileMask[3, .., TensorIndex.Slice(4, 7), TensorIndex.Slice(5, 8)] = 1;
        var pairs = new LocalizationPairs(tensor(new long[] { 0, 1, 2, 3, 4, 5, 1, 2 }, [4, 2]), tensor(new[] { 1f, -1f, 1f, -1f, -1f, 1f, 1f, -1f }, [4, 2]),
            ones(4, 2), zeros(4, 2), 4f);
        var onCanvas = zeros(4, dtype: ScalarType.Int64);

        var whole = SigmoidContrastiveLoss.ComputeLocalizedPairs(canvas.Select(onCanvas), tags, pairs, 0.25f, tileMask, null, 2f, 0.05f);
        var wholeGradients = Gradients(whole, leaves);
        var (tiles, cropMask) = Mosaic.TileCrops(canvas, tileMask);
        var cropped = SigmoidContrastiveLoss.ComputeLocalizedPairs(tiles, tags, pairs, 0.25f, cropMask, null, 2f, 0.05f);
        var croppedGradients = Gradients(cropped, leaves);

        Assert.Equal(new long[] { 4, EmbeddingDim, 2, 2 }, tiles.Coarse.shape);
        Assert.Equal(whole.item<float>(), cropped.item<float>(), 4);
        for (var i = 0; i < leaves.Length; i++)
            Assert.True(croppedGradients[i].allclose(wholeGradients[i], rtol: 1e-3, atol: 1e-5), $"gradient {i}");
    }

    [Fact]
    public void ForwardPooledFine_MatchesForwardPooled()
    {
        var tower = Tower();
        tower.eval();
        tower.Fine = tower.NewFineLocalization();
        using (no_grad())
            tower.Fine.Projection.weight!.normal_(0, 0.1);
        var input = rand(2, 3, 64, 64);
        var mask = Upsample2x(CoarseMask(2, 64 / 4));

        var (pooled, localization) = tower.ForwardPooled(input, mask);
        var (finePooled, parts) = tower.ForwardPooledFine(input, mask);

        Assert.True(finePooled.allclose(pooled, rtol: 1e-4, atol: 1e-5));
        Assert.True(parts.Dense().allclose(localization, rtol: 1e-4, atol: 1e-5));
    }

    [Fact]
    public void TheZeroProjection_StillGetsAGradient()
    {
        var tower = Tower();
        tower.Fine = tower.NewFineLocalization();
        var (_, localization) = tower.ForwardPooled(rand(1, 3, 64, 64), ones(1, 1, 32, 32));
        localization.pow(2).sum().backward();
        Assert.True(tower.Fine.Projection.weight!.grad!.abs().sum().item<float>() > 0);
    }

    [Fact]
    public void NeedsEveryStageFusedIntoAFusedOnlyMap()
    {
        var partial = new ImageTower(EmbeddingDim, 8, [8, 12, 16], [1, 1, 1], fusedStages: 1) { LocalizationFromFusedOnly = true };
        Assert.Throws<InvalidOperationException>(() => partial.NewFineLocalization());
        var full = new ImageTower(EmbeddingDim, 8, [8, 12, 16], [1, 1, 1], fusedStages: 2);
        Assert.Throws<InvalidOperationException>(() => full.NewFineLocalization());
    }

    [Fact]
    public void SaveThenLoad_RoundTripsTheLevel()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var tower = Tower();
            tower.Fine = tower.NewFineLocalization();
            using (no_grad())
                tower.Fine.Projection.weight!.normal_(0, 0.1);
            var vocabulary = TagVocabulary.CreateEmpty();
            vocabulary.AddTag("solo");
            var embeddings = TagEmbeddingStore.CreateEmpty(EmbeddingDim);
            embeddings.AppendRow(new float[EmbeddingDim]);

            Checkpoint.Save(directory, tower, ModelConfig.From(tower, 64), vocabulary, embeddings, new LogitScaleBias());
            Assert.True(Checkpoint.HasFineLocalization(directory));
            var (loaded, _, _, _, _) = Checkpoint.Load(directory);

            Assert.NotNull(loaded.Fine);
            Assert.Equal(2, loaded.OutputStride);
            Assert.True(loaded.Fine!.Projection.weight!.allclose(tower.Fine.Projection.weight!));

            tower.Fine = null;
            var plain = Directory.CreateTempSubdirectory().FullName;
            try
            {
                Checkpoint.Save(plain, tower, ModelConfig.From(tower, 64), vocabulary, embeddings, new LogitScaleBias());
                Assert.Null(Checkpoint.Load(plain).ImageTower.Fine);
            }
            finally
            {
                Directory.Delete(plain, recursive: true);
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
