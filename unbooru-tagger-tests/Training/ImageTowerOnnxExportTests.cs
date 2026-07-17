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
    public void ExportedGraph_ProducesSameOutputsAsTorchSharpForward()
    {
        const int embeddingDim = 8;
        const int inputSize = 32;

        var tower = new ImageTower(embeddingDim, stemChannels: 8, stageChannels: [8, 16], blocksPerStage: [1, 1]);
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
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }
}
