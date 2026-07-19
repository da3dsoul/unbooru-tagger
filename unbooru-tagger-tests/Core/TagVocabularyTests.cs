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
    public void SaveDelta_PersistsOnlyNewTagsWithoutRewritingTheBaseSnapshot()
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        vocabulary.AddTag("solo");

        var basePath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.json");
        var deltaPath = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.jsonl");
        try
        {
            vocabulary.Save(basePath);
            var baseWriteTime = File.GetLastWriteTimeUtc(basePath);

            // A tag reappearing (no new row) shouldn't touch the delta file at all.
            vocabulary.TryGet("solo", out var solo);
            solo.ImageCount++;
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
