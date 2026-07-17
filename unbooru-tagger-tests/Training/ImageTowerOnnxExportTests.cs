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
        manual_seed(42);
        const int embeddingDim = 8;
        const int inputSize = 32;

        var tower = new ImageTower(embeddingDim, stemChannels: 4, stageChannels: [4, 8], blocksPerStage: [1, 1]);
        tower.eval();

        var input = rand([1, 3, inputSize, inputSize]);
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
