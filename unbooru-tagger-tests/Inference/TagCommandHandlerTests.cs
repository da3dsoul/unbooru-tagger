using SkiaSharp;
using UnbooruTagger.Core.Embedding;
using UnbooruTagger.Core.Runtime;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Inference.Commands;
using UnbooruTagger.Training.Export;
using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Inference;

public class TagCommandHandlerTests
{
    [Fact]
    public void Score_ReturnsOneEntryPerVocabularyTagAboveThreshold()
    {
        const int embeddingDim = 6;
        const int inputSize = 32;

        var modelDir = Directory.CreateTempSubdirectory().FullName;
        var imagePath = CreateTestImage(inputSize);
        try
        {
            var tower = new ImageTower(embeddingDim, stemChannels: 4, stageChannels: [4], blocksPerStage: [1]);
            tower.eval();
            FillDeterministic(tower);
            ImageTowerOnnxExporter.Export(tower, Path.Combine(modelDir, ModelBundle.ImageEncoderFileName), inputSize);

            var vocabulary = TagVocabulary.CreateEmpty();
            vocabulary.AddTag("solo");
            vocabulary.AddTag("1girl");
            vocabulary.Save(Path.Combine(modelDir, ModelBundle.VocabularyFileName));

            var embeddings = TagEmbeddingStore.CreateEmpty(embeddingDim);
            embeddings.AppendRow(new float[embeddingDim]);
            embeddings.AppendRow(new float[embeddingDim]);
            embeddings.Save(Path.Combine(modelDir, ModelBundle.EmbeddingsFileName));

            using var model = ModelBundle.Load(modelDir, inputSize);
            var scores = TagCommandHandler.Score(model, imagePath, threshold: -10f);

            Assert.Equal(2, scores.Count);
            Assert.Contains(scores, r => r.Tag == "solo");
            Assert.Contains(scores, r => r.Tag == "1girl");
        }
        finally
        {
            File.Delete(imagePath);
            Directory.Delete(modelDir, recursive: true);
        }
    }

    /// <summary>
    /// Deliberately NOT random init — see ImageTowerOnnxExportTests's comment: some
    /// random seeds land GroupNorm's pre-normalization variance close enough to zero (a
    /// float32 precision cliff) that run-to-run kernel nondeterminism (confirmed not
    /// fixed by manual_seed) occasionally tips sqrt(variance + eps) into NaN/Infinity,
    /// which then propagates through every tag's score and made this test flaky through
    /// no fault of TagCommandHandler itself. Fixed, varying-but-bounded-away-from-zero
    /// values sidestep the cliff entirely.
    /// </summary>
    private static void FillDeterministic(ImageTower tower)
    {
        using (no_grad())
        {
            foreach (var parameter in tower.parameters())
            {
                var values = Enumerable.Range(0, (int)parameter.numel()).Select(i => 0.01f * ((i % 7) - 3)).ToArray();
                parameter.copy_(tensor(values, parameter.shape));
            }
        }
    }

    private static string CreateTestImage(int size)
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.png");
        using var bitmap = new SKBitmap(size, size);
        // Two colors, not a flat fill: a perfectly uniform image can drive GroupNorm's
        // pre-normalization variance to exactly (or near) zero, which is a real,
        // documented precision cliff (see ImageTowerOnnxExportTests's comment) that can
        // tip sqrt(variance + eps) into NaN depending on the exact resampled pixel
        // values -- unrelated to what this test actually checks.
        using (var canvas = new SKCanvas(bitmap))
        {
            canvas.Clear(SKColors.CornflowerBlue);
            canvas.DrawRect(new SKRect(0, 0, size / 2f, size / 2f), new SKPaint { Color = SKColors.Salmon });
        }
        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        using var stream = File.Create(path);
        data.SaveTo(stream);
        return path;
    }
}
