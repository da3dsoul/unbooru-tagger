using System.Text.Json;
using UnbooruTagger.Core.Scoring;
using UnbooruTagger.Training.Checkpoints;
using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class ClassificationPoolingTests
{
    // One image, a 32x32 map of 2-dim features: everything (1, 0) except one 2x2 corner of (0, 4), the "glove".
    private static Tensor Map()
    {
        var values = new float[2 * 32 * 32];
        for (var y = 0; y < 32; y++)
            for (var x = 0; x < 32; x++)
            {
                var glove = y < 2 && x < 2;
                values[y * 32 + x] = glove ? 0f : 1f;
                values[32 * 32 + y * 32 + x] = glove ? 4f : 0f;
            }
        return tensor(values, [1, 2, 32, 32]);
    }

    private static readonly Tensor Tags = tensor(new[] { 1f, 0f, 0f, 1f }, [2, 2]); // row 0 = "background", row 1 = "glove"

    private static float[] Scores(Tensor map, Tensor mask, float temperature, long[] rows)
    {
        var pooled = ImageTower.MaskedPool(map, mask);
        var (cells, valid) = ClassificationPooling.Cells(map, mask);
        return ClassificationPooling.RawScores(pooled, cells, valid, Tags, tensor(rows), temperature).data<float>().ToArray();
    }

    [Fact]
    public void ASmallObject_CarriesThePooledScore_WhereTheAverageBarelyMoves()
    {
        using var mask = ones(1, 1, 32, 32);
        var averaged = Scores(Map(), mask, 0.25f, []);
        var pooled = Scores(Map(), mask, 0.25f, [1]);
        Assert.Equal(4f * 4 / 1024, averaged[1], 4);   // the glove is 4 of 1024 locations
        Assert.True(pooled[1] > 0.5f, $"pooled glove score {pooled[1]}");
        Assert.Equal(averaged[0], pooled[0], 5);        // rows not pooled keep the average
    }

    [Fact]
    public void AHighTemperature_GivesBackTheAverage()
    {
        using var mask = ones(1, 1, 32, 32);
        var averaged = Scores(Map(), mask, 0.25f, []);
        var pooled = Scores(Map(), mask, 1000f, [0, 1]);
        Assert.Equal(averaged[0], pooled[0], 3);
        Assert.Equal(averaged[1], pooled[1], 3);
    }

    [Fact]
    public void LetterboxPadding_IsNotACell()
    {
        // Content only in the bottom half: the glove corner is padding and must not count.
        var maskValues = new float[32 * 32];
        Array.Fill(maskValues, 1f, 16 * 32, 16 * 32);
        using var mask = tensor(maskValues, [1, 1, 32, 32]);
        var pooled = Scores(Map(), mask, 0.25f, [1]);
        Assert.Equal(0f, pooled[1], 4);
    }

    [Fact]
    public void DenseLogits_MatchLinearLogits_ForTheSameScores()
    {
        var pooled = new[] { 1f, 2f, -1f, 0.5f, 0f, 3f };        // 3 images x 2 dims
        var tags = new[] { 1f, 0f, 0.5f, -1f };                  // 2 tags
        var raw = new float[6];
        for (var i = 0; i < 3; i++)
            for (var t = 0; t < 2; t++)
                raw[i * 2 + t] = pooled[i * 2] * tags[t * 2] + pooled[i * 2 + 1] * tags[t * 2 + 1];
        var scaling = new LogitScaling(3f, -1f);
        using var linear = ImageTagLogits.Linear(pooled, tags, 2, scaling);
        using var dense = ImageTagLogits.Dense(raw, 3, scaling);
        for (var i = 0; i < 3; i++)
            for (var t = 0; t < 2; t++)
                Assert.Equal(linear.Logit(i, t), dense.Logit(i, t), 5);
        Assert.Equal(linear.Columns([1], CPU).data<float>().ToArray(), dense.Columns([1], CPU).data<float>().ToArray());
        Assert.Equal(linear.Rows(1, 2, CPU).data<float>().ToArray(), dense.Rows(1, 2, CPU).data<float>().ToArray());
    }

    [Fact]
    public void TrainedRows_ArePositivesCertainNegativesAndTheTopScoringOthers()
    {
        // 1 image, 4 tags of 2 dims, all pooled: 0 = positive, 1 = full-strength negative, 2 and 3 unverified
        // negatives, 3 scoring higher on the map (it matches the glove corner, 2 matches nothing).
        using var mask = ones(1, 1, 32, 32);
        var (cells, valid) = ClassificationPooling.Cells(Map(), mask);
        using var tags = tensor(new[] { 1f, 0f, 1f, 0f, -1f, 0f, 0f, 1f }, [4, 2]);
        using var targets = new PairTargets(
            tensor(new[] { 1f, -1f, -1f, -1f }, [1, 4]), tensor(new[] { 1f, 1f, 1f, 1f }, [1, 4]), tensor(new[] { 0f, 0f, 1f, 1f }, [1, 4]), 1f);
        Assert.Equal([0L, 1, 3], ClassificationPooling.TrainedRows(cells, valid, tags, [0, 1, 2, 3], targets, 0.25f, minedPerImage: 1).data<long>().ToArray());
        Assert.Equal([0L, 1], ClassificationPooling.TrainedRows(cells, valid, tags, [0, 1, 2, 3], targets, 0.25f, minedPerImage: 0).data<long>().ToArray());
    }

    [Fact]
    public void TheSchedule_AnnealsLogLinearly_FromWhenPoolingWasTurnedOn()
    {
        var schedule = new ClassificationPooling.Schedule(StartStep: 1000, StartTemperature: 8f, TargetTemperature: 0.5f, AnnealSteps: 400);
        Assert.Equal(8f, schedule.At(500), 4);    // before the start: still the start temperature
        Assert.Equal(8f, schedule.At(1000), 4);
        Assert.Equal(4f, schedule.At(1100), 4);   // a quarter of the way: one halving of four
        Assert.Equal(2f, schedule.At(1200), 4);
        Assert.Equal(0.5f, schedule.At(1400), 4);
        Assert.Equal(0.5f, schedule.At(9000), 4); // held at the target afterwards
        Assert.Equal(0.5f, (schedule with { AnnealSteps = 0 }).At(1000), 4);
    }

    [Fact]
    public void TheSchedule_SurvivesTheModelConfig()
    {
        var config = new ModelConfig(512, 512, 96, [96], [3], ClassificationPoolingTemperature: 2f,
            ClassificationPoolingSchedule: new ClassificationPooling.Schedule(271_754, 8f, 0.25f, 12_000));
        var loaded = JsonSerializer.Deserialize<ModelConfig>(JsonSerializer.Serialize(config))!;
        Assert.Equal(config.ClassificationPoolingSchedule, loaded.ClassificationPoolingSchedule);
        Assert.Null(JsonSerializer.Deserialize<ModelConfig>("""{"EmbeddingDim":512,"InputSize":512,"StemChannels":96,"StageChannels":[96],"BlocksPerStage":[3]}""")!.ClassificationPoolingSchedule);
    }
}
