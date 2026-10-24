using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class CropConsistencyTests
{
    [Fact]
    public void SampleWindows_AreSquare_AndStayInsideTheContent()
    {
        // 64px canvas, 8x8 grid (stride 8); content is the middle 64x32 band (rows 2..5).
        var boxes = Enumerable.Repeat(new LetterboxBox(0, 16, 64, 32), 200).ToList();
        var windows = CropConsistency.SampleWindows(boxes, 64, 8, 8, new Random(1));

        foreach (var w in windows)
        {
            Assert.InRange(w.Size, 1, 4);
            Assert.InRange(w.Y, 2, 6 - w.Size);
            Assert.InRange(w.X, 0, 8 - w.Size);
        }
    }

    [Fact]
    public void CropPixels_TheWholeGrid_IsTheImage_AndAHalfWindowIsItsQuadrant()
    {
        var pixels = rand(1, 3, 16, 16);
        Assert.True(CropConsistency.CropPixels(pixels, [new CellWindow(0, 0, 4)], 4).allclose(pixels));

        // Bottom-right quadrant of a horizontal ramp (value = column), upsampled 2x: it spans
        // columns 8..15, so the crop's values run from ~7.75 to ~15.25.
        var ramp = arange(16, ScalarType.Float32).reshape(1, 1, 1, 16).expand(1, 3, 16, 16).contiguous();
        var crop = CropConsistency.CropPixels(ramp, [new CellWindow(2, 2, 2)], 4);
        Assert.Equal(16, crop.shape[3]);
        Assert.InRange(crop.min().item<float>(), 7.5f, 8.1f);
        Assert.InRange(crop.max().item<float>(), 14.9f, 15.5f);
    }

    [Fact]
    public void WindowMask_MarksExactlyTheWindow()
    {
        var mask = CropConsistency.WindowMask([new CellWindow(1, 2, 2)], 4, 4);
        Assert.Equal(4f, mask.sum().item<float>());
        Assert.Equal(1f, mask[0, 0, 1, 2].item<float>());
        Assert.Equal(0f, mask[0, 0, 0, 0].item<float>());
    }

    [Fact]
    public void WindowConsistency_RewardsHeatWhereTheTeacherFindsTheTag_AndIgnoresUnsureTeachers()
    {
        // 2x2 map, one positive pair; the tag scores high only at the top-left cell.
        using var map = tensor(new[] { 4f, -4f, -4f, -4f }, [1, 1, 2, 2]).requires_grad_(true);
        using var tags = tensor(new[] { 1f }, [1, 1]);
        using var pairs = new LocalizationPairs(
            tensor(new long[] { 0 }, [1, 1]), tensor(new[] { 1f }, [1, 1]), tensor(new[] { 1f }, [1, 1]), tensor(new[] { 0f }, [1, 1]), 1f);
        var sure = tensor(new[] { 6f }, [1, 1]);

        float Loss(CellWindow window, Tensor teacher) =>
            SigmoidContrastiveLoss.ComputeWindowConsistency(map, tags, pairs, CropConsistency.WindowMask([window], 2, 2), teacher, 1f, null).item<float>();

        // The teacher finds the tag in the crop: a window over the hot cell agrees, one over a cold cell doesn't.
        Assert.True(Loss(new CellWindow(0, 0, 1), sure) < Loss(new CellWindow(1, 1, 1), sure));
        // A teacher at p = 0.5 teaches nothing.
        Assert.Equal(0f, Loss(new CellWindow(1, 1, 1), tensor(new[] { 0f }, [1, 1])), 5);

        // Gradient reaches the map only inside the window.
        var loss = SigmoidContrastiveLoss.ComputeWindowConsistency(map, tags, pairs, CropConsistency.WindowMask([new CellWindow(1, 1, 1)], 2, 2), sure, 1f, null);
        loss.backward();
        var grad = map.grad!.data<float>().ToArray();
        Assert.Equal(0f, grad[0]);
        Assert.True(grad[3] < 0); // raising the cold cell inside the window lowers the loss
    }

    [Fact]
    public void WindowConsistency_SkipsUntrustedTags_AndUnsureAnswersWhenGated()
    {
        using var map = tensor(new[] { 4f, -4f, -4f, -4f }, [1, 1, 2, 2]);
        using var tags = tensor(new[] { 1f }, [1, 1]);
        using var pairs = new LocalizationPairs(
            tensor(new long[] { 0 }, [1, 1]), tensor(new[] { 1f }, [1, 1]), tensor(new[] { 1f }, [1, 1]), tensor(new[] { 0f }, [1, 1]), 1f);
        using var window = CropConsistency.WindowMask([new CellWindow(1, 1, 1)], 2, 2);

        float Loss(float teacherLogit, float trusted, float confidentAbove) =>
            SigmoidContrastiveLoss.ComputeWindowConsistency(map, tags, pairs, window, tensor(new[] { teacherLogit }, [1, 1]), 1f, null,
                tensor(new[] { trusted }, [1, 1]), confidentAbove).item<float>();

        Assert.True(Loss(6f, 1f, 0.9f) > 0);           // trusted tag, confident teacher: trains
        Assert.Equal(0f, Loss(6f, 0f, 0.9f), 6);       // untrusted tag: nothing
        Assert.Equal(0f, Loss(1f, 1f, 0.9f), 6);       // p = 0.73 is not confident enough: nothing
    }

    [Fact]
    public void TeacherGate_TrustsOnlyWellLearnedLocalizableTagsWithEnoughPositives()
    {
        var vocabulary = UnbooruTagger.Core.Vocabulary.TagVocabulary.CreateEmpty();
        foreach (var tag in new[] { "hat", "solo", "rare_hat", "weak_hat" })
            vocabulary.AddTag(tag);
        var report = new EvaluationReport(100, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        [
            new TagEvaluation("hat", 50, 10, 10, 0.9, 0.9, 0.9, 0.5f, 0.9),
            new TagEvaluation("solo", 50, 10, 10, 0.95, 0.9, 0.9, 0.5f, 0.9),     // not localizable
            new TagEvaluation("rare_hat", 3, 10, 10, 1.0, 1.0, 1.0, 0.5f, 1.0),  // too few positives
            new TagEvaluation("weak_hat", 50, 10, 10, 0.6, 0.6, 0.6, 0.5f, 0.6), // not learned yet
        ]);

        var gate = CropTeacherGate.From(report, vocabulary, [true, false, true, true], 0.8, 20);
        Assert.Equal([true, false, false, false], gate.Trusted);
        Assert.Equal(1, gate.TrustedCount);
    }
}
