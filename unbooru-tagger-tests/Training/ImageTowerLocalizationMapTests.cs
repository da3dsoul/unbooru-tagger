using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace UnbooruTagger.Tests.Training;

public class ImageTowerLocalizationMapTests
{
    [Fact]
    public void LocalMap_IsTheFusedChainWithoutTheCoarsestLateral()
    {
        var tower = new ImageTower(8, stemChannels: 8, stageChannels: [8, 12, 16, 20], blocksPerStage: [1, 1, 1, 1], fusedStages: 3) { LocalizationFromFusedOnly = true };
        tower.eval();
        using var input = randn(2, 3, 64, 64);
        using var _ = no_grad();

        var fused = tower.ForwardMaps(input).Localization;
        tower.LocalizationSkippedLaterals = 1;
        var (spatial, local) = tower.ForwardMaps(input);
        tower.LocalizationSkippedLaterals = 0;

        // Same stride-4 resolution and the same classification map, different localization map.
        Assert.Equal(fused.shape, local.shape);
        Assert.True(spatial.allclose(tower.ForwardMaps(input).Spatial));
        Assert.False(local.allclose(fused));

        // fused = up(up(L0) + L1) + L2 and local = up(L1) + L2, so their difference is L0 upsampled
        // twice: constant over each 4x4 block of the stride-4 map.
        var difference = fused - local;
        var blocks = functional.interpolate(functional.avg_pool2d(difference, 4), scale_factor: [4.0, 4.0], mode: InterpolationMode.Nearest);
        Assert.True(difference.allclose(blocks, rtol: 1e-4, atol: 1e-5));
    }

    [Fact]
    public void LocalMap_NeedsAFusedOnlyMapThatKeepsALateral()
    {
        var tower = new ImageTower(8, stemChannels: 8, stageChannels: [8, 12, 16], blocksPerStage: [1, 1, 1], fusedStages: 1);
        Assert.Throws<InvalidOperationException>(() => tower.LocalizationSkippedLaterals = 1);
        tower.LocalizationFromFusedOnly = true;
        Assert.Throws<InvalidOperationException>(() => tower.LocalizationSkippedLaterals = 1);
    }

    [Theory]
    [InlineData("earrings", 10, 12, "size: small <32px")]
    [InlineData("shoes", 40, 30, "size: medium 32-96px")]
    [InlineData("shoes", 200, 150, "size: large >=96px")]
    [InlineData("character:hatsune_miku", 10, 10, "size: characters")]
    public void SizeBucket_UsesTheLargestNonIgnoredBox(string tag, int width, int height, string expected)
    {
        var instances = new List<GoldInstance>
        {
            new(tag, [0, 0, width, height], GoldKind.Certain),
            new(tag, [0, 0, 400, 400], GoldKind.Ignore),
        };
        Assert.Equal(expected, LocalizationEvaluator.SizeBucket(tag, instances));
    }
}
