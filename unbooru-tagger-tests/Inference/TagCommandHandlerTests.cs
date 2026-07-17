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
        manual_seed(7);
        const int embeddingDim = 6;
        const int inputSize = 32;

        var modelDir = Directory.CreateTempSubdirectory().FullName;
        var imagePath = CreateTestImage(inputSize);
        try
        {
            var tower = new ImageTower(embeddingDim, stemChannels: 4, stageChannels: [4], blocksPerStage: [1]);
            tower.eval();
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

    private static string CreateTestImage(int size)
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.png");
        using var bitmap = new SKBitmap(size, size);
        using (var canvas = new SKCanvas(bitmap))
            canvas.Clear(SKColors.CornflowerBlue);
        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(SKEncodedImageFormat.Png, 100);
        using var stream = File.Create(path);
        data.SaveTo(stream);
        return path;
    }
}
