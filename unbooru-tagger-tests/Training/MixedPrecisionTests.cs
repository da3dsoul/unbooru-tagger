using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

/// <summary>
/// <see cref="ImageTower.ComputeType"/> (train --bf16): the tower computes in another dtype from fp32 weights cast in
/// the graph, and hands back fp32 maps and fp32 gradients.
/// </summary>
public class MixedPrecisionTests
{
    private static ImageTower Tower()
    {
        var tower = new ImageTower(8, stemChannels: 8, stageChannels: [8, 12, 16, 20], blocksPerStage: [1, 1, 2, 1], fusedStages: 3) { LocalizationFromFusedOnly = true };
        // LayerScale starts near 0, which would leave the blocks out of the comparison.
        using (no_grad())
            foreach (var (name, parameter) in tower.named_parameters())
                if (name.EndsWith("LayerScale"))
                    parameter.fill_(1);
        return tower;
    }

    private static Tensor Mask()
    {
        using var mask = ones(3, 1, 16, 16);
        return mask.index_fill(3, arange(10, 16), 0); // the right 6 columns are letterbox padding
    }

    private static (Tensor Pooled, Tensor Localization, Tensor Spatial, Tensor[] Gradients) Run(ImageTower tower, Tensor images, Tensor mask)
    {
        tower.zero_grad();
        var (pooled, localization) = tower.ForwardPooled(images, mask);
        var (spatial, _) = tower.ForwardMaps(images);
        (pooled.square().sum() + localization.abs().mean() + spatial.square().mean()).backward();
        return (pooled, localization, spatial, tower.parameters().Select(p => p.grad!.clone()).ToArray());
    }

    // Float64 runs exactly the cast path bf16 does (conv/norm through functional ops with weights cast per call,
    // LayerScale cast, input cast, maps cast back), but precisely enough to compare with fp32 to fp32 tolerances.
    [Fact]
    public void ComputingInAnotherDtype_MatchesTheFp32Tower()
    {
        var tower = Tower();
        using var images = randn(3, 3, 64, 64);
        using var mask = Mask();

        var expected = Run(tower, images, mask);
        tower.ComputeType = ScalarType.Float64;
        var actual = Run(tower, images, mask);

        Assert.Equal(ScalarType.Float32, actual.Pooled.dtype);
        Assert.Equal(ScalarType.Float32, actual.Localization.dtype);
        Assert.Equal(ScalarType.Float32, actual.Spatial.dtype);
        Assert.True(actual.Pooled.allclose(expected.Pooled, rtol: 1e-4, atol: 1e-5), "pooled differs");
        Assert.True(actual.Localization.allclose(expected.Localization, rtol: 1e-4, atol: 1e-5), "localization differs");
        Assert.True(actual.Spatial.allclose(expected.Spatial, rtol: 1e-4, atol: 1e-5), "spatial differs");
        Assert.All(tower.parameters(), p => Assert.Equal(ScalarType.Float32, p.dtype));
        for (var i = 0; i < expected.Gradients.Length; i++)
        {
            Assert.Equal(ScalarType.Float32, actual.Gradients[i].dtype);
            Assert.True(actual.Gradients[i].allclose(expected.Gradients[i], rtol: 1e-3, atol: 1e-6), $"parameter {i} differs");
        }
    }

    [Fact]
    public void Bf16_StaysCloseToFp32()
    {
        var tower = Tower();
        using var images = randn(3, 3, 64, 64);
        using var mask = Mask();

        var expected = Run(tower, images, mask);
        tower.ComputeType = ScalarType.BFloat16;
        var actual = Run(tower, images, mask);

        // bf16 keeps 8 bits of mantissa: a few percent of the output's scale, not an exact match.
        static double RelativeError(Tensor actual, Tensor expected) =>
            (actual - expected).norm().item<float>() / expected.norm().item<float>();
        Assert.Equal(ScalarType.Float32, actual.Spatial.dtype);
        Assert.InRange(RelativeError(actual.Pooled, expected.Pooled), 0, 0.05);
        Assert.InRange(RelativeError(actual.Localization, expected.Localization), 0, 0.05);
        Assert.InRange(RelativeError(actual.Spatial, expected.Spatial), 0, 0.05);
        for (var i = 0; i < expected.Gradients.Length; i++)
        {
            Assert.Equal(ScalarType.Float32, actual.Gradients[i].dtype);
            Assert.InRange(RelativeError(actual.Gradients[i], expected.Gradients[i]), 0, 0.1);
        }
    }
}
