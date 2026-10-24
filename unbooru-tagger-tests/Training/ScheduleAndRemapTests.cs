using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class ScheduleAndRemapTests
{
    [Fact]
    public void Schedule_WarmsUpLinearly_ThenDecaysToTheFloor()
    {
        Assert.Equal(0.1, LearningRateSchedule.At(0, 1000, 1.0, 10), 6);
        Assert.Equal(1.0, LearningRateSchedule.At(9, 1000, 1.0, 10), 6);
        Assert.Equal(1.0, LearningRateSchedule.At(10, 1000, 1.0, 10), 6);
        Assert.Equal(0.525, LearningRateSchedule.At(505, 1000, 1.0, 10), 3); // halfway through the decay
        Assert.Equal(0.05, LearningRateSchedule.At(1000, 1000, 1.0, 10), 6);
        Assert.Equal(0.05, LearningRateSchedule.At(5000, 1000, 1.0, 10), 6); // past the end stays at the floor
    }

    [Fact]
    public void Compact_DropsRowsNoImageCarries_AndApplyRemapsByName()
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        foreach (var tag in new[] { "long hair", "long_hair", "hat", "stale" })
            vocabulary.AddTag(tag);
        IReadOnlyList<IReadOnlyList<int>> rows = [[1, 2], [1], [2]];

        var compact = VocabularyRemap.Compact(vocabulary, rows);
        Assert.Equal(["long_hair", "hat"], compact.Records.Select(r => r.Tag));
        Assert.Equal([2, 2], compact.Records.Select(r => r.ImageCount));

        var (remapped, unmatched) = VocabularyRemap.Apply(vocabulary, rows, compact);
        Assert.Equal([[0, 1], [0], [1]], remapped.Select(r => r.ToArray()));
        Assert.Equal(2, unmatched);
    }
}
