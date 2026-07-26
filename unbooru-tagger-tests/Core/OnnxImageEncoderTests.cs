using SkiaSharp;
using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Training.Export;
using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Core;

/// <summary>
/// <see cref="OnnxImageEncoder"/> recomputes the pooled embedding itself from the
/// graph's spatial output instead of trusting the exported graph's own (unmasked)
/// pooled_embedding output — see the "fix the onnx model" follow-up: training masks out
/// letterbox padding when pooling, so inference needs to match or the two disagree on
/// what a non-square image's embedding should be.
/// </summary>
public class OnnxImageEncoderTests
{
    [Fact]
    public void Encode_PooledEmbeddingExcludesLetterboxPadding()
    {
        manual_seed(11);
        const int embeddingDim = 4;
        const int inputSize = 16;

        var tower = new ImageTower(embeddingDim, stemChannels: 4, stageChannels: [4], blocksPerStage: [1]);
        tower.eval();
        using (no_grad())
        {
            foreach (var parameter in tower.parameters())
            {
                var values = Enumerable.Range(0, (int)parameter.numel()).Select(i => 0.01f * ((i % 7) - 3)).ToArray();
                parameter.copy_(tensor(values, parameter.shape));
            }
        }

        var tempFile = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.onnx");
        var imagePath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.png");
        try
        {
            ImageTowerOnnxExporter.Export(tower, tempFile, inputSize);

            // Wide, non-square source: letterboxes into top/bottom padding bars once
            // decoded at inputSize x inputSize, so this actually exercises masking.
            using (var bitmap = new SKBitmap(new SKImageInfo(32, 8, SKColorType.Rgba8888, SKAlphaType.Unpremul)))
            {
                using (var canvas = new SKCanvas(bitmap))
                {
                    canvas.Clear(SKColors.CornflowerBlue);
                    canvas.DrawRect(new SKRect(0, 0, 16, 4), new SKPaint { Color = SKColors.Salmon });
                }
                using var image = SKImage.FromBitmap(bitmap);
                using var data = image.Encode(SKEncodedImageFormat.Png, 100);
                using var stream = File.Create(imagePath);
                data.SaveTo(stream);
            }

            using var encoder = new OnnxImageEncoder(tempFile, embeddingDim, inputSize);
            var encoding = encoder.Encode(imagePath);

            // Content should show real top/bottom padding for a 32x8 source in a 16x16 canvas.
            Assert.True(encoding.Content.Y > 0);
            Assert.True(encoding.Content.Height < inputSize);

            var manualMaskedPooled = ManualMaskedPool(encoding.SpatialFeatures, encoding.Content, inputSize);
            var manualUnmaskedPooled = ManualUnmaskedPool(encoding.SpatialFeatures);

            for (var c = 0; c < embeddingDim; c++)
                Assert.Equal(manualMaskedPooled[c], encoding.PooledEmbedding[c], 4);

            // The masked and plain-average pools should actually differ here -- otherwise
            // this test wouldn't be exercising anything padding-related.
            Assert.True(
                MathF.Abs(manualUnmaskedPooled[0] - manualMaskedPooled[0]) > 1e-5f,
                $"Expected masked ({manualMaskedPooled[0]}) and unmasked ({manualUnmaskedPooled[0]}) pools to differ.");
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
            if (File.Exists(imagePath))
                File.Delete(imagePath);
        }
    }

    private static float[] ManualMaskedPool(SpatialFeatureMap spatial, LetterboxBox content, int canvasSize)
    {
        var validity = ImagePreprocessing.ComputeSpatialValidity(content, canvasSize, spatial.Height, spatial.Width);
        var sum = new float[spatial.Channels];
        var count = 0;
        for (var y = 0; y < spatial.Height; y++)
        for (var x = 0; x < spatial.Width; x++)
        {
            if (!validity[y, x])
                continue;
            var vector = spatial[y, x];
            for (var c = 0; c < vector.Length; c++)
                sum[c] += vector[c];
            count++;
        }
        for (var c = 0; c < sum.Length; c++)
            sum[c] /= count;
        return sum;
    }

    private static float[] ManualUnmaskedPool(SpatialFeatureMap spatial)
    {
        var sum = new float[spatial.Channels];
        for (var y = 0; y < spatial.Height; y++)
        for (var x = 0; x < spatial.Width; x++)
        {
            var vector = spatial[y, x];
            for (var c = 0; c < vector.Length; c++)
                sum[c] += vector[c];
        }
        for (var c = 0; c < sum.Length; c++)
            sum[c] /= spatial.Height * spatial.Width;
        return sum;
    }
}
