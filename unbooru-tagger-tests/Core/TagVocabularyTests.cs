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
}
