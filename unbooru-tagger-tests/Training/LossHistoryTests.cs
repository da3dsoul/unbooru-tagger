using UnbooruTagger.Training.Commands;

namespace UnbooruTagger.Tests.Training;

public class LossHistoryTests
{
    [Fact]
    public void Buckets_AreMeansOverFixedGlobalSteps()
    {
        var history = new LossHistory(bucketSteps: 4);
        // Starting mid-bucket (a resume at step 2): the first bucket covers steps 3-4 only.
        for (var step = 3; step <= 10; step++)
            history.Add(step, step);
        Assert.Equal(new[] { (4L, 3.5f), (8L, 6.5f) }, history.Buckets);
    }

    [Fact]
    public void Load_RestoresTheHistory_CutBackToTheResumedStep()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var history = new LossHistory(bucketSteps: 4);
            for (var step = 1; step <= 10; step++)
                history.Add(step, step);
            history.Save(directory);

            // Resumed exactly where it was saved: the unfinished bucket (steps 9-10) carries on.
            var resumed = LossHistory.Load(directory, resumedStep: 10);
            Assert.Equal(4, resumed.BucketSteps);
            Assert.Equal(new[] { (4L, 2.5f), (8L, 6.5f) }, resumed.Buckets);
            resumed.Add(11, 11);
            resumed.Add(12, 12);
            Assert.Equal((12L, 10.5f), resumed.Buckets[^1]);

            // Resumed from an older checkpoint: later buckets go, the unfinished one too.
            var older = LossHistory.Load(directory, resumedStep: 6);
            Assert.Equal(new[] { (4L, 2.5f) }, older.Buckets);
            older.Add(7, 7);
            older.Add(8, 9);
            Assert.Equal((8L, 8f), older.Buckets[^1]);

            Assert.Empty(LossHistory.Load(Path.Combine(directory, "missing"), 10).Buckets);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
