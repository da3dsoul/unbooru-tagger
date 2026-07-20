using SkiaSharp;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Training.Checkpoints;
using UnbooruTagger.Training.Commands;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class TrainCommandHandlerTests
{
    [Fact]
    public void Run_ResumesEpochCountAndOptimizerStateAcrossInvocations()
    {
        manual_seed(7);
        const int embeddingDim = 4;
        const int inputSize = 32;

        var checkpointDir = Directory.CreateTempSubdirectory().FullName;
        var manifestDir = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var manifestPath = CreateManifest(manifestDir, inputSize);

            var firstResult = TrainCommandHandler.Run(
                manifestPath, cacheDir: null, checkpointDir, embeddingDim, inputSize,
                epochs: 1, batchSize: 1, learningRate: 1e-3, validationFraction: 0.25, earlyStoppingPatience: 3);
            Assert.Equal(0, firstResult);

            Assert.True(Checkpoint.Exists(checkpointDir));
            Assert.True(TrainingState.Exists(checkpointDir));
            Assert.Equal(1, TrainingState.LoadProgress(checkpointDir).CompletedEpochs);

            // Re-running with the same --epochs the checkpoint already reached should be
            // a no-op (nothing left to train), not silently restart from epoch 0 -- assert
            // via the training-state file's write time, since CompletedEpochs alone can't
            // distinguish "skipped" from "ran one epoch and landed on the same number".
            var progressPath = Directory.GetFiles(checkpointDir, "training_progress.json").Single();
            var writeTimeAfterFirstRun = File.GetLastWriteTimeUtc(progressPath);

            var secondResult = TrainCommandHandler.Run(
                manifestPath, cacheDir: null, checkpointDir, embeddingDim, inputSize,
                epochs: 1, batchSize: 1, learningRate: 1e-3, validationFraction: 0.25, earlyStoppingPatience: 3);
            Assert.Equal(0, secondResult);
            Assert.Equal(writeTimeAfterFirstRun, File.GetLastWriteTimeUtc(progressPath));
            Assert.Equal(1, TrainingState.LoadProgress(checkpointDir).CompletedEpochs);

            // Raising --epochs should resume from epoch 1 (not restart at 0) and load the
            // saved optimizer state without throwing -- the real risk in this whole
            // feature, since TorchSharp's load_state_dict requires the resumed optimizer
            // to have been constructed over parameters with matching shapes/order.
            var thirdResult = TrainCommandHandler.Run(
                manifestPath, cacheDir: null, checkpointDir, embeddingDim, inputSize,
                epochs: 2, batchSize: 1, learningRate: 1e-3, validationFraction: 0.25, earlyStoppingPatience: 3);
            Assert.Equal(0, thirdResult);
            Assert.Equal(2, TrainingState.LoadProgress(checkpointDir).CompletedEpochs);
        }
        finally
        {
            Directory.Delete(checkpointDir, recursive: true);
            Directory.Delete(manifestDir, recursive: true);
        }
    }

    private static string CreateManifest(string directory, int inputSize)
    {
        var tagSets = new[]
        {
            new[] { "solo" },
            new[] { "1girl" },
            new[] { "solo", "1girl" },
            new[] { "twintails" },
        };

        var entries = new List<DatasetImageEntry>();
        for (var i = 0; i < tagSets.Length; i++)
        {
            var imagePath = Path.Combine(directory, $"{i}.png");
            using var bitmap = new SKBitmap(inputSize, inputSize);
            using (var canvas = new SKCanvas(bitmap))
                canvas.Clear(new SKColor((byte)(i * 40), (byte)(255 - i * 40), 128));
            using var image = SKImage.FromBitmap(bitmap);
            using var data = image.Encode(SKEncodedImageFormat.Png, 100);
            using (var stream = File.Create(imagePath))
                data.SaveTo(stream);

            entries.Add(new DatasetImageEntry(imagePath, tagSets[i]));
        }

        var manifestPath = Path.Combine(directory, "manifest.json");
        new DatasetManifest(entries).Save(manifestPath);
        return manifestPath;
    }
}
