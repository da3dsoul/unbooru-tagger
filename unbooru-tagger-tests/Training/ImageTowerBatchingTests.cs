using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

/// <summary>
/// The trainer runs the mosaic canvases in the batch's own forward pass (rows after the images) instead of a second
/// pass. That is only the same computation if every op in the tower treats each image on its own.
/// </summary>
public class ImageTowerBatchingTests
{
    private static ImageTower Tower()
    {
        var tower = new ImageTower(8, stemChannels: 8, stageChannels: [8, 12, 16, 20], blocksPerStage: [1, 1, 2, 1], fusedStages: 3) { LocalizationFromFusedOnly = true };
        // LayerScale starts near 0, which would leave the blocks out of both comparisons.
        var scaled = 0;
        using (no_grad())
            foreach (var (name, parameter) in tower.named_parameters())
                if (name.EndsWith("LayerScale"))
                {
                    parameter.fill_(1);
                    scaled++;
                }
        Assert.Equal(5, scaled); // one per block
        return tower;
    }

    [Fact]
    public void OneForwardOfImagesAndCanvases_EqualsTwoSeparateForwards()
    {
        var tower = Tower();
        using var images = randn(3, 3, 64, 64);
        using var canvases = randn(2, 3, 64, 64);
        using var _ = no_grad();

        var (spatial, localization) = tower.ForwardMaps(cat([images, canvases], 0));
        var (imageSpatial, imageLocalization) = tower.ForwardMaps(images);
        var (canvasSpatial, canvasLocalization) = tower.ForwardMaps(canvases);

        Assert.True(spatial.narrow(0, 0, 3).allclose(imageSpatial, rtol: 1e-4, atol: 1e-5));
        Assert.True(spatial.narrow(0, 3, 2).allclose(canvasSpatial, rtol: 1e-4, atol: 1e-5));
        Assert.True(localization.narrow(0, 0, 3).allclose(imageLocalization, rtol: 1e-4, atol: 1e-5));
        Assert.True(localization.narrow(0, 3, 2).allclose(canvasLocalization, rtol: 1e-4, atol: 1e-5));
    }

    [Fact]
    public void OneForwardOfImagesAndCanvases_GivesTheSameGradients()
    {
        var tower = Tower();
        using var images = randn(3, 3, 64, 64);
        using var canvases = randn(2, 3, 64, 64);

        Tensor[] Gradients(Func<Tensor> loss)
        {
            tower.zero_grad();
            loss().backward();
            return tower.parameters().Select(p => p.grad!.clone()).ToArray();
        }

        // A loss using both parts' maps, as the step's classification and mosaic terms do.
        var merged = Gradients(() =>
        {
            var (spatial, localization) = tower.ForwardMaps(cat([images, canvases], 0));
            return spatial.narrow(0, 0, 3).square().mean() + localization.narrow(0, 3, 2).abs().mean();
        });
        var separate = Gradients(() => tower.ForwardMaps(images).Spatial.square().mean() + tower.ForwardMaps(canvases).Localization.abs().mean());

        Assert.Equal(separate.Length, merged.Length);
        for (var i = 0; i < merged.Length; i++)
            Assert.True(merged[i].allclose(separate[i], rtol: 1e-3, atol: 1e-6), $"parameter {i} differs");
    }

    // A letterbox-style mask at the full map's resolution (16x16 for 64px at stride 4): image 0 all content, image 1
    // the left 10 columns, image 2 the top 5 rows (not a multiple of any block, so blocks are partly covered).
    private static Tensor Mask()
    {
        var values = new float[3 * 16 * 16];
        for (var y = 0; y < 16; y++)
            for (var x = 0; x < 16; x++)
            {
                values[y * 16 + x] = 1;
                values[256 + y * 16 + x] = x < 10 ? 1 : 0;
                values[512 + y * 16 + x] = y < 5 ? 1 : 0;
            }
        return tensor(values, [3, 1, 16, 16]);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)] // the full map is the localization map: falls back to building it
    public void ForwardPooled_EqualsPoolingTheFullMap(bool fusedOnly)
    {
        var tower = Tower();
        tower.LocalizationFromFusedOnly = fusedOnly;
        using var images = randn(3, 3, 64, 64);
        using var mask = Mask();

        Tensor[] Gradients(Func<(Tensor Pooled, Tensor Localization)> forward, out Tensor pooled, out Tensor localization)
        {
            tower.zero_grad();
            (pooled, localization) = forward();
            (pooled.square().sum() + localization.abs().mean()).backward();
            return tower.parameters().Select(p => p.grad!.clone()).ToArray();
        }

        var direct = Gradients(() => tower.ForwardPooled(images, mask), out var pooled, out var localization);
        var viaFullMap = Gradients(() =>
        {
            var (spatial, map) = tower.ForwardMaps(images);
            return (ImageTower.MaskedPool(spatial, mask), map);
        }, out var expectedPooled, out var expectedLocalization);

        Assert.Equal(16, tower.ForwardMaps(images).Spatial.shape[2]);
        Assert.True(pooled.allclose(expectedPooled, rtol: 1e-4, atol: 1e-5), "pooled differs");
        Assert.True(localization.allclose(expectedLocalization, rtol: 1e-4, atol: 1e-5), "localization differs");
        for (var i = 0; i < direct.Length; i++)
            Assert.True(direct[i].allclose(viaFullMap[i], rtol: 1e-3, atol: 1e-6), $"parameter {i} differs");
    }
}
