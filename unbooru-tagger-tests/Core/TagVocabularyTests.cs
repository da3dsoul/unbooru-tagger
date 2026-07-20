using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Tests.Core;

public class TagVocabularyTests
{
    [Fact]
    public void PromoteIfThresholdMet_OnlyPromotesOnceImageCountReachesThreshold()
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        var record = vocabulary.AddTag("new_tag");
        record.ImageCount = 5;

        vocabulary.PromoteIfThresholdMet("new_tag", minImageThreshold: 10);
        Assert.Equal(TagStatus.WarmStartOnly, record.Status);

        record.ImageCount = 10;
        vocabulary.PromoteIfThresholdMet("new_tag", minImageThreshold: 10);
        Assert.Equal(TagStatus.Trained, record.Status);
    }

    [Fact]
    public void SaveAndLoad_RoundTripsRecords()
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        vocabulary.AddTag("solo");
        vocabulary.AddTag("1girl");

        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.json");
        try
        {
            vocabulary.Save(path);
            var loaded = TagVocabulary.Load(path);

            Assert.Equal(2, loaded.Records.Count);
            Assert.True(loaded.TryGet("solo", out var record));
            Assert.Equal(0, record.RowIndex);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void AddTag_ThrowsForDuplicateTag()
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        vocabulary.AddTag("solo");

        Assert.Throws<InvalidOperationException>(() => vocabulary.AddTag("solo"));
    }

    [Fact]
    public void SaveDelta_OnlyTouchesTheFileWhenSomethingActuallyChanged()
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        vocabulary.AddTag("solo");

        var basePath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.json");
        var deltaPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.jsonl");
        try
        {
            vocabulary.Save(basePath);
            var baseWriteTime = File.GetLastWriteTimeUtc(basePath);

            // Nothing changed since the last Save -- no delta file needed yet.
            vocabulary.SaveDelta(deltaPath);
            Assert.False(File.Exists(deltaPath));

            vocabulary.AddTag("1girl");
            vocabulary.SaveDelta(deltaPath);
            Assert.True(File.Exists(deltaPath));

            // The base snapshot itself is untouched by SaveDelta -- only the small delta
            // file grows, which is the whole point: checkpoint cost shouldn't scale with
            // vocabulary size.
            Assert.Equal(baseWriteTime, File.GetLastWriteTimeUtc(basePath));

            var loaded = TagVocabulary.Load(basePath, deltaPath);
            Assert.Equal(2, loaded.Records.Count);
            Assert.True(loaded.TryGet("1girl", out var record));
            Assert.Equal(1, record.RowIndex);
        }
        finally
        {
            File.Delete(basePath);
            File.Delete(deltaPath);
        }
    }

    [Fact]
    public void RecordObservation_MarksExistingTagsDirtySoTheirImageCountSurvivesInTheDelta()
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        vocabulary.AddTag("solo");

        var basePath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.json");
        var deltaPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.jsonl");
        try
        {
            vocabulary.Save(basePath);

            // A tag reappearing via RecordObservation (no new row) still needs its
            // updated image count checkpointed -- the delta is the only durable record
            // of that update until the next full compaction.
            vocabulary.RecordObservation("solo");
            vocabulary.RecordObservation("solo");
            vocabulary.SaveDelta(deltaPath);
            Assert.True(File.Exists(deltaPath));

            var loaded = TagVocabulary.Load(basePath, deltaPath);
            Assert.True(loaded.TryGet("solo", out var record));
            Assert.Equal(2, record.ImageCount);
        }
        finally
        {
            File.Delete(basePath);
            File.Delete(deltaPath);
        }
    }

    [Fact]
    public void SaveDelta_LaterCheckpointsForTheSameTagOverrideEarlierOnesOnLoad()
    {
        var vocabulary = TagVocabulary.CreateEmpty();

        var basePath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.json");
        var deltaPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.jsonl");
        try
        {
            vocabulary.Save(basePath); // empty base snapshot

            vocabulary.RecordObservation("solo"); // new row, count 1
            vocabulary.SaveDelta(deltaPath); // checkpoint after "page 1"

            vocabulary.RecordObservation("solo"); // same row, count 2
            vocabulary.SaveDelta(deltaPath); // checkpoint after "page 2"

            var loaded = TagVocabulary.Load(basePath, deltaPath);
            Assert.Single(loaded.Records);
            Assert.True(loaded.TryGet("solo", out var record));
            Assert.Equal(0, record.RowIndex);
            Assert.Equal(2, record.ImageCount);
        }
        finally
        {
            File.Delete(basePath);
            File.Delete(deltaPath);
        }
    }

    [Fact]
    public void SaveDelta_IsIdempotentAcrossMultipleCheckpoints()
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        vocabulary.AddTag("solo");

        var basePath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.json");
        var deltaPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.jsonl");
        try
        {
            vocabulary.Save(basePath);

            vocabulary.AddTag("1girl");
            vocabulary.SaveDelta(deltaPath); // checkpoint after "page 1"

            vocabulary.AddTag("twintails");
            vocabulary.SaveDelta(deltaPath); // checkpoint after "page 2" -- must not re-append 1girl

            var loaded = TagVocabulary.Load(basePath, deltaPath);
            Assert.Equal(3, loaded.Records.Count);
            Assert.True(loaded.TryGet("twintails", out var record));
            Assert.Equal(2, record.RowIndex);
        }
        finally
        {
            File.Delete(basePath);
            File.Delete(deltaPath);
        }
    }
}
