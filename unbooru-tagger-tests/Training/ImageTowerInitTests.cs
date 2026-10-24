using TorchSharp.Modules;
using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class ImageTowerInitTests
{
    [Fact]
    public void FreshTower_HasIdentityNorms_AndAFiniteForwardPass()
    {
        // TorchSharp's GroupNorm leaves weight/bias uninitialized; several towers make a
        // garbage (often NaN on CPU) allocation very likely to show up if the explicit init regresses.
        for (var attempt = 0; attempt < 5; attempt++)
        {
            var tower = new ImageTower(16, stemChannels: 32, stageChannels: [32, 64, 96], blocksPerStage: [2, 2, 2], fusedStages: 1);
            var norms = tower.modules().OfType<GroupNorm>().ToList();
            Assert.NotEmpty(norms);
            foreach (var norm in norms)
            {
                Assert.True(norm.weight!.eq(1).all().item<bool>());
                Assert.True(norm.bias!.eq(0).all().item<bool>());
            }

            var (pooled, spatial) = tower.forward(rand(2, 3, 64, 64));
            Assert.False(pooled.isnan().any().item<bool>());
            Assert.False(spatial.isnan().any().item<bool>());
        }
    }

    [Fact]
    public void FusedOnlyLocalizationMap_LeavesClassificationUnchanged_AndDropsTheLastStage()
    {
        var tower = new ImageTower(8, stemChannels: 8, stageChannels: [8, 12, 16], blocksPerStage: [1, 1, 1], fusedStages: 2);
        tower.eval();
        var input = rand(1, 3, 64, 64);

        var (pooled, spatial) = tower.forward(input);
        var (sameSpatial, sameLocalization) = tower.ForwardMaps(input);
        Assert.True(sameLocalization.allclose(sameSpatial));

        tower.LocalizationFromFusedOnly = true;
        var (fusedPooled, fusedSpatial) = tower.forward(input);
        var (_, localization) = tower.ForwardMaps(input);
        Assert.True(fusedPooled.allclose(pooled));
        Assert.True(fusedSpatial.allclose(spatial));

        // Full map = localization map + the last stage's projection, nearest-upsampled.
        var lastStage = tower.Projection.forward(LastStage(tower, input));
        var upsampled = nn.functional.interpolate(lastStage, size: [spatial.shape[2], spatial.shape[3]], mode: InterpolationMode.Nearest);
        Assert.True((localization + upsampled).allclose(spatial, rtol: 1e-4, atol: 1e-5));
    }

    [Fact]
    public void FusedOnlyLocalizationMap_RequiresFusion() =>
        Assert.Throws<InvalidOperationException>(() => new ImageTower(8, stemChannels: 8, stageChannels: [8, 16], blocksPerStage: [1, 1]) { LocalizationFromFusedOnly = true });

    [Theory]
    [InlineData(8, new[] { 8, 16 }, 0)]
    [InlineData(8, new[] { 8, 12, 16 }, 2)]
    [InlineData(16, new[] { 8, 12, 16 }, 1)]
    public void OutputStride_MatchesTheSpatialMap(int stemChannels, int[] stageChannels, int fusedStages)
    {
        var tower = new ImageTower(8, stemChannels, stageChannels, stageChannels.Select(_ => 1).ToArray(), fusedStages: fusedStages);
        var (_, spatial) = tower.forward(rand(1, 3, 64, 64));
        Assert.Equal(64 / tower.OutputStride, spatial.shape[2]);
    }

    private static Tensor LastStage(ImageTower tower, Tensor input)
    {
        var x = tower.Stem.forward(input);
        foreach (var layer in tower.Layers)
            x = layer.forward(x);
        return x;
    }
}
