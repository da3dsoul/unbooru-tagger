using SkiaSharp;
using UnbooruTagger.Core.Embedding;
using UnbooruTagger.Core.Runtime;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Inference.Commands;
using UnbooruTagger.Training.Export;
using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Inference;

public class DetectCommandHandlerTests
{
    [Fact]
    public void Detect_ReturnsBoxesWithinImageBoundsForTagsAboveThreshold()
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
            var detections = DetectCommandHandler.Detect(model, imagePath, threshold: -10f, boxThreshold: -10f, boxPercentile: 0f);

            Assert.Equal(2, detections.Count);
            foreach (var detection in detections)
            {
                Assert.NotEmpty(detection.Boxes);
                foreach (var box in detection.Boxes)
                {
                    Assert.InRange(box.X, 0, inputSize);
                    Assert.InRange(box.Y, 0, inputSize);
                    Assert.True(box.X + box.Width <= inputSize);
                    Assert.True(box.Y + box.Height <= inputSize);
                }
            }
        }
        finally
        {
            File.Delete(imagePath);
            Directory.Delete(modelDir, recursive: true);
        }
    }

    [Fact]
    public void Run_WithOutOption_WritesAnAnnotatedPngTheSameSizeAsTheInput()
    {
        manual_seed(7);
        const int embeddingDim = 6;
        const int inputSize = 224; // Run() loads the model with ModelBundle's default input size.

        var modelDir = Directory.CreateTempSubdirectory().FullName;
        var imagePath = CreateTestImage(inputSize);
        var outputPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.png");
        try
        {
            var tower = new ImageTower(embeddingDim, stemChannels: 4, stageChannels: [4], blocksPerStage: [1]);
            tower.eval();
            ImageTowerOnnxExporter.Export(tower, Path.Combine(modelDir, ModelBundle.ImageEncoderFileName), inputSize);

            var vocabulary = TagVocabulary.CreateEmpty();
            vocabulary.AddTag("solo");
            vocabulary.Save(Path.Combine(modelDir, ModelBundle.VocabularyFileName));

            var embeddings = TagEmbeddingStore.CreateEmpty(embeddingDim);
            embeddings.AppendRow(new float[embeddingDim]);
            embeddings.Save(Path.Combine(modelDir, ModelBundle.EmbeddingsFileName));

            var exitCode = DetectCommandHandler.Run(modelDir, imagePath, threshold: -10f, boxThreshold: -10f, boxPercentile: 0f, outputPath);

            Assert.Equal(0, exitCode);
            Assert.True(File.Exists(outputPath));

            using var annotated = SKBitmap.Decode(outputPath);
            Assert.Equal(inputSize, annotated.Width);
            Assert.Equal(inputSize, annotated.Height);
        }
        finally
        {
            File.Delete(imagePath);
            if (File.Exists(outputPath))
                File.Delete(outputPath);
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
