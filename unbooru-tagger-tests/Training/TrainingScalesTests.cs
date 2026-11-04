using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class TrainingScalesTests
{
    [Fact]
    public void Parse_ReadsSizeBatchShareAndStoredShare()
    {
        var scales = TrainingScales.Parse("512:10:0.3, 1024:4:0.5:0.25,2048:1:0.2:1");

        Assert.Equal([new TrainingScale(512, 10, 0.3), new TrainingScale(1024, 4, 0.5, 0.25), new TrainingScale(2048, 1, 0.2, 1)], scales);
        Assert.Throws<ArgumentException>(() => TrainingScales.Parse("512:10"));
        Assert.Throws<ArgumentException>(() => new TrainingScales(TrainingScales.Parse("512:10:1,512:4:1"), 0));
        Assert.Throws<ArgumentException>(() => new TrainingScales(TrainingScales.Parse("512:10:1:1.5"), 0));
    }

    [Fact]
    public void Next_DrawsEachScaleByItsShare_AndStoredCountAveragesTheStoredShare()
    {
        var scales = new TrainingScales(TrainingScales.Parse("512:10:3,1024:4:5:0.3,2048:1:2:1"), seed: 7);
        const int draws = 100_000;
        var counts = new Dictionary<int, int>();
        var stored = 0;
        for (var i = 0; i < draws; i++)
        {
            var scale = scales.Next();
            counts[scale.Size] = counts.GetValueOrDefault(scale.Size) + 1;
            if (scale.Size == 1024)
                stored += scales.StoredCount(scale);
        }

        Assert.InRange(counts[512] / (double)draws, 0.29, 0.31);
        Assert.InRange(counts[1024] / (double)draws, 0.49, 0.51);
        Assert.InRange(counts[2048] / (double)draws, 0.19, 0.21);
        Assert.InRange(stored / (double)counts[1024], 1.17, 1.23); // 0.3 x 4
        Assert.Equal(1, scales.StoredCount(scales.Scales[2]));
    }
}
