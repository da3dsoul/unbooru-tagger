using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

/// <summary>
/// Regression test for a real bug found in production training: without LayerScale,
/// a stack of ConvNeXtBlocks with ordinary random initialization could blow up into
/// NaN on the very first forward pass (an unlucky near-zero GroupNorm variance
/// compounding through several stacked blocks). LayerScale keeps each block near-identity
/// at init, so this checks that a realistically-deep/wide tower never produces NaN,
/// across many seeds, with no special-cased weights.
/// </summary>
public class ImageTowerStabilityTests
{
    [Fact]
    public void Forward_NeverProducesNaN_AcrossManySeedsWithOrdinaryRandomInit()
    {
        for (var seed = 0; seed < 15; seed++)
        {
            manual_seed(seed);
            var tower = new ImageTower(embeddingDim: 64, stemChannels: 32, stageChannels: [32, 64, 128], blocksPerStage: [2, 2, 2]);
            tower.eval();

            var input = randn([2, 3, 64, 64]);
            var (pooled, spatial) = tower.forward(input);

            Assert.False(pooled.data<float>().Any(float.IsNaN), $"pooled output was NaN for seed {seed}");
            Assert.False(spatial.data<float>().Any(float.IsNaN), $"spatial output was NaN for seed {seed}");
        }
    }
}
