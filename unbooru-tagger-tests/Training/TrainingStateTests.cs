using UnbooruTagger.Training.Checkpoints;

namespace UnbooruTagger.Tests.Training;

public class TrainingStateTests
{
    [Fact]
    public void LoadProgress_RoundTripsMidEpochFields()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var progress = new TrainingProgress(
                CompletedEpochs: 2,
                EarlyStoppingBestLoss: 0.5,
                EarlyStoppingEvaluationsSinceImprovement: 1,
                CompletedStepsInEpoch: 37,
                ValidationSplitSeed: 12345);

            File.WriteAllText(
                Path.Combine(directory, "training_progress.json"),
                System.Text.Json.JsonSerializer.Serialize(progress));

            var loaded = TrainingState.LoadProgress(directory);

            Assert.Equal(progress, loaded);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void LoadProgress_DefaultsMidEpochFieldsForOlderCheckpoints()
    {
        // A checkpoint written before mid-epoch resume existed has no
        // CompletedStepsInEpoch/ValidationSplitSeed properties at all -- this must still
        // load instead of throwing, landing on an epoch boundary (step 0) with a
        // re-derivable seed rather than failing to resume.
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            File.WriteAllText(
                Path.Combine(directory, "training_progress.json"),
                """{"CompletedEpochs":3,"EarlyStoppingBestLoss":0.25,"EarlyStoppingEvaluationsSinceImprovement":0}""");

            var loaded = TrainingState.LoadProgress(directory);

            Assert.Equal(3, loaded.CompletedEpochs);
            Assert.Equal(0, loaded.CompletedStepsInEpoch);
            Assert.Equal(0, loaded.ValidationSplitSeed);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
