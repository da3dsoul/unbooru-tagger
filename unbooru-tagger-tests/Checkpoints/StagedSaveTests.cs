using UnbooruTagger.Training.Checkpoints;

namespace UnbooruTagger.Tests.Checkpoints;

public class StagedSaveTests
{
    [Fact]
    public void Run_ReplacesEveryFile_AndLeavesNoStagingBehind()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            File.WriteAllText(Path.Combine(directory, "image_tower.dat"), "old weights");
            File.WriteAllText(Path.Combine(directory, "training_progress.json"), "old progress");
            File.WriteAllText(Path.Combine(directory, "vocabulary.txt"), "kept");

            StagedSave.Run(directory, staging =>
            {
                File.WriteAllText(Path.Combine(staging, "image_tower.dat"), "new weights");
                File.WriteAllText(Path.Combine(staging, "training_progress.json"), "new progress");
            });

            Assert.Equal("new weights", File.ReadAllText(Path.Combine(directory, "image_tower.dat")));
            Assert.Equal("new progress", File.ReadAllText(Path.Combine(directory, "training_progress.json")));
            Assert.Equal("kept", File.ReadAllText(Path.Combine(directory, "vocabulary.txt")));
            Assert.False(Directory.Exists(Path.Combine(directory, StagedSave.DirectoryName)));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Run_ThatDiesPartway_LeavesThePreviousCheckpointWhole()
    {
        // A device fault while the optimizer state is copied off the device: the weights are already written.
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            File.WriteAllText(Path.Combine(directory, "image_tower.dat"), "old weights");
            File.WriteAllText(Path.Combine(directory, "optimizer.dat"), "old optimizer");

            Assert.Throws<InvalidOperationException>(() => StagedSave.Run(directory, staging =>
            {
                File.WriteAllText(Path.Combine(staging, "image_tower.dat"), "new weights");
                throw new InvalidOperationException("device lost");
            }));

            Assert.Equal("old weights", File.ReadAllText(Path.Combine(directory, "image_tower.dat")));
            Assert.Equal("old optimizer", File.ReadAllText(Path.Combine(directory, "optimizer.dat")));

            // The next save discards what the failed one left.
            StagedSave.Run(directory, staging => File.WriteAllText(Path.Combine(staging, "optimizer.dat"), "new optimizer"));
            Assert.Equal("old weights", File.ReadAllText(Path.Combine(directory, "image_tower.dat")));
            Assert.Equal("new optimizer", File.ReadAllText(Path.Combine(directory, "optimizer.dat")));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
