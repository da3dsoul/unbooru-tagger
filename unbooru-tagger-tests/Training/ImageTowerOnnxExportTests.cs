using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnbooruTagger.Training.Export;
using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

/// <summary>
/// TorchSharp has no ONNX exporter, so <see cref="ImageTowerOnnxExporter"/> hand-builds
/// the graph node-for-node. That's the riskiest hand-written code in this scaffold —
/// this test is the thing that actually proves the exported graph computes the same
/// thing the TorchSharp module does, not just that it loads.
/// </summary>
public class ImageTowerOnnxExportTests
{
    [Fact]
    public void ExportedGraph_ProducesSameOutputsAsTorchSharpForward() =>
        AssertExportMatches(new ImageTower(8, stemChannels: 8, stageChannels: [8, 16], blocksPerStage: [1, 1]), 32, expectedSpatialSize: 4);

    [Fact]
    public void ExportedGraph_WithMultiScaleFusion_ProducesSameOutputsAsTorchSharpForward() =>
        // Stem /4 -> 16x16 (stage 0), /8 -> 8x8 (stage 1), /16 -> 4x4 (stage 2); fusing 2
        // stages brings the spatial output back up to stage 0's 16x16.
        AssertExportMatches(new ImageTower(8, stemChannels: 8, stageChannels: [8, 12, 16], blocksPerStage: [1, 1, 1], fusedStages: 2), 64, expectedSpatialSize: 16);

    [Fact]
    public void ExportedGraph_WithFusedOnlyLocalizationMap_ExportsItAsAThirdOutput() =>
        AssertExportMatches(
            new ImageTower(8, stemChannels: 8, stageChannels: [8, 12, 16], blocksPerStage: [1, 1, 1], fusedStages: 2) { LocalizationFromFusedOnly = true },
            64, expectedSpatialSize: 16);

    [Fact]
    public void ExportedGraph_RunsABatchAsIfEachImageWereAlone()
    {
        const int inputSize = 64;
        var tower = new ImageTower(8, stemChannels: 8, stageChannels: [8, 12, 16], blocksPerStage: [1, 1, 1], fusedStages: 2) { LocalizationFromFusedOnly = true };
        tower.eval();
        var tempFile = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.onnx");
        try
        {
            ImageTowerOnnxExporter.Export(tower, tempFile, inputSize);
            using var session = new InferenceSession(tempFile);
            Assert.Equal(-1, session.InputMetadata["pixel_values"].Dimensions[0]);

            // Two different images: GroupNorm's statistics must stay per image, not per batch.
            var imageSize = 3 * inputSize * inputSize;
            var first = Enumerable.Range(0, imageSize).Select(i => 0.01f * ((i % 11) - 5)).ToArray();
            var second = Enumerable.Range(0, imageSize).Select(i => 0.03f * ((i % 5) - 2) + 0.2f).ToArray();
            float[][] Run(float[] pixels, int batch)
            {
                using var results = session.Run([NamedOnnxValue.CreateFromTensor("pixel_values", new DenseTensor<float>(pixels, [batch, 3, inputSize, inputSize]))]);
                return new[] { "pooled_embedding", "spatial_features", ImageTowerOnnxExporter.LocalizationOutputName }
                    .Select(name => results.First(r => r.Name == name).AsTensor<float>().ToArray()).ToArray();
            }

            var batched = Run([.. first, .. second], 2);
            var alone = new[] { Run(first, 1), Run(second, 1) };
            for (var output = 0; output < batched.Length; output++)
            {
                var perImage = batched[output].Length / 2;
                for (var image = 0; image < 2; image++)
                {
                    Assert.Equal(perImage, alone[image][output].Length);
                    for (var i = 0; i < perImage; i++)
                        Assert.Equal(alone[image][output][i], batched[output][image * perImage + i], 4);
                }
            }
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Fact]
    public void ExportedGraph_WithLocalMap_LeavesOutTheCoarsestLateral()
    {
        var tower = new ImageTower(8, stemChannels: 8, stageChannels: [8, 12, 16, 20], blocksPerStage: [1, 1, 1, 1], fusedStages: 3) { LocalizationFromFusedOnly = true };
        tower.LocalizationSkippedLaterals = 1;
        AssertExportMatches(tower, 64, expectedSpatialSize: 16);
    }

    private static void AssertExportMatches(ImageTower tower, int inputSize, int expectedSpatialSize)
    {
        tower.eval();

        // Deliberately NOT random: this was originally seeded random init, but for some
        // seeds the pre-normalization variance inside GroupNorm lands close enough to
        // zero (a float32 precision cliff edge) that tiny run-to-run nondeterminism in
        // libtorch's CPU kernels (not fixed by manual_seed, set_num_threads(1), or a
        // wider network — all tried and confirmed insufficient) tips sqrt(variance+eps)
        // into occasional NaN/overflow, making the test flaky through no fault of the
        // export logic itself. Fixed, varying-but-bounded-away-from-zero values sidestep
        // the cliff entirely and make this purely a correctness check of the exporter.
        using (no_grad())
        {
            foreach (var parameter in tower.parameters())
            {
                var values = Enumerable.Range(0, (int)parameter.numel()).Select(i => 0.01f * ((i % 7) - 3)).ToArray();
                parameter.copy_(tensor(values, parameter.shape));
            }
        }

        var input = tensor(
            Enumerable.Range(0, 3 * inputSize * inputSize).Select(i => 0.01f * ((i % 11) - 5)).ToArray(),
            [1, 3, inputSize, inputSize]);
        var (expectedPooled, expectedSpatial) = tower.forward(input);
        Assert.Equal(expectedSpatialSize, expectedSpatial.shape[2]);

        var tempFile = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.onnx");
        try
        {
            ImageTowerOnnxExporter.Export(tower, tempFile, inputSize);

            using var session = new InferenceSession(tempFile);
            var inputTensor = new DenseTensor<float>(input.data<float>().ToArray(), [1, 3, inputSize, inputSize]);
            using var results = session.Run([NamedOnnxValue.CreateFromTensor("pixel_values", inputTensor)]);

            var actualPooled = results.First(r => r.Name == "pooled_embedding").AsTensor<float>().ToArray();
            var actualSpatial = results.First(r => r.Name == "spatial_features").AsTensor<float>().ToArray();

            var expectedPooledArray = expectedPooled.data<float>().ToArray();
            var expectedSpatialArray = expectedSpatial.data<float>().ToArray();

            Assert.Equal(expectedPooledArray.Length, actualPooled.Length);
            for (var i = 0; i < expectedPooledArray.Length; i++)
                Assert.Equal(expectedPooledArray[i], actualPooled[i], 3);

            Assert.Equal(expectedSpatialArray.Length, actualSpatial.Length);
            for (var i = 0; i < expectedSpatialArray.Length; i++)
                Assert.Equal(expectedSpatialArray[i], actualSpatial[i], 3);

            var actualLocalization = results.FirstOrDefault(r => r.Name == ImageTowerOnnxExporter.LocalizationOutputName);
            if (!tower.LocalizationFromFusedOnly)
            {
                Assert.Null(actualLocalization);
                return;
            }

            var expectedLocalization = tower.ForwardMaps(input).Localization.data<float>().ToArray();
            var actualLocalizationArray = actualLocalization!.AsTensor<float>().ToArray();
            Assert.Equal(expectedLocalization.Length, actualLocalizationArray.Length);
            for (var i = 0; i < expectedLocalization.Length; i++)
                Assert.Equal(expectedLocalization[i], actualLocalizationArray[i], 3);
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }
}
