using TorchSharp;
using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class LocalityNegativesTests
{
    // Tags 0 and 1 are one-hot embeddings; window k's glimpse embedding says how strongly each is there.
    private static readonly Tensor Tags = eye(4);

    private static GlimpseChoice Windows(params (int X, int Y)[] corners) => new(
        zeros(1, corners.Length, dtype: ScalarType.Int64), // 32px windows: close looks
        tensor(corners.Select(c => (long)c.X).ToArray()).reshape(1, -1),
        tensor(corners.Select(c => (long)c.Y).ToArray()).reshape(1, -1),
        ones(1, corners.Length, dtype: ScalarType.Bool));

    private static Tensor Cells(params float[][] windows) =>
        tensor(windows.SelectMany(w => w).ToArray(), [windows.Length, 4]).t().unsqueeze(0).contiguous(); // [1, D, K]

    private static (Tensor Rows, Tensor Valid) BothTags() =>
        (tensor(new long[] { 0, 1 }).reshape(1, 2), ones(1, 2, dtype: ScalarType.Bool));

    [Fact]
    public void TagsFoundFarApart_LearnTheyAreNeverTogether_ThenTrainEachOtherNoInTheirWindows()
    {
        using var locality = new LocalityNegatives(CPU, slotBits: 10) { MinObservations = 0.5f };
        using var scaling = new LogitScaleBias(initialScale: 1, initialBias: 0);
        using var choice = Windows((0, 0), (400, 400));
        var cells = Cells([5, 0, 0, 0], [0, 5, 0, 0]).requires_grad_(true);
        var (rows, valid) = BothTags();

        // The first look only records the pair: nothing is known about it yet.
        Assert.Equal(0, locality.Step(cells, choice, rows, valid, Tags, scaling).item<float>(), 5);
        Assert.Equal(0, locality.Together(0, 1)!.Value, 5);

        // Then each is trained "no" in the other's window: B's logit there is 0, softplus(0) = ln 2, times the confidence in A.
        var loss = locality.Step(cells, choice, rows, valid, Tags, scaling);
        var confidence = 1 / (1 + Math.Exp(-5));
        Assert.Equal(2 * Math.Log(2) * confidence, loss.item<float>(), 3);
        loss.backward();
        // The gradient lowers tag 1's evidence in tag 0's window, and tag 0's in tag 1's.
        var grad = cells.grad!;
        Assert.True(grad[0, 1, 0].item<float>() > 0);
        Assert.True(grad[0, 0, 1].item<float>() > 0);
    }

    [Fact]
    public void PerPair_AveragesOverAnImagesPairs_AndTheGate_DropsUnsureCalls()
    {
        using var scaling = new LogitScaleBias(initialScale: 1, initialBias: 0);
        using var choice = Windows((0, 0), (400, 400));
        var cells = Cells([5, 0, 0, 0], [0, 5, 0, 0]);
        var (rows, valid) = BothTags();
        var confidence = 1 / (1 + Math.Exp(-5)); // 0.993

        using var perPair = new LocalityNegatives(CPU, slotBits: 10) { MinObservations = 0.5f, PerPair = true };
        perPair.Step(cells, choice, rows, valid, Tags, scaling);
        Assert.Equal(Math.Log(2) * confidence, perPair.Step(cells, choice, rows, valid, Tags, scaling).item<float>(), 3);

        using var gated = new LocalityNegatives(CPU, slotBits: 10) { MinObservations = 0.5f, ConfidenceGate = 0.999f };
        gated.Step(cells, choice, rows, valid, Tags, scaling);
        Assert.Equal(0, gated.Together(0, 1)!.Value, 5); // the table still learns from the call
        Assert.Equal(0, gated.Step(cells, choice, rows, valid, Tags, scaling).item<float>(), 5);
    }

    [Fact]
    public void DetachScaling_LeavesTheScaleAndBiasUntrained()
    {
        using var locality = new LocalityNegatives(CPU, slotBits: 10) { MinObservations = 0.5f, DetachScaling = true };
        using var scaling = new LogitScaleBias(initialScale: 1, initialBias: 0);
        using var choice = Windows((0, 0), (400, 400));
        var cells = Cells([5, 0, 0, 0], [0, 5, 0, 0]).requires_grad_(true);
        var (rows, valid) = BothTags();

        locality.Step(cells, choice, rows, valid, Tags, scaling);
        var loss = locality.Step(cells, choice, rows, valid, Tags, scaling);
        Assert.True(loss.item<float>() > 0);
        loss.backward();
        Assert.All(scaling.parameters(), p => Assert.True(p.grad is null || p.grad.abs().sum().item<float>() == 0));
        Assert.True(cells.grad![0, 1, 0].item<float>() > 0);
    }

    [Fact]
    public void TagsFoundTogether_GiveNoNegatives()
    {
        using var locality = new LocalityNegatives(CPU, slotBits: 10) { MinObservations = 0.5f };
        using var scaling = new LogitScaleBias(initialScale: 1, initialBias: 0);
        using var choice = Windows((100, 100), (120, 110)); // centres 20px apart
        var cells = Cells([5, 0, 0, 0], [0, 5, 0, 0]);
        var (rows, valid) = BothTags();

        locality.Step(cells, choice, rows, valid, Tags, scaling);
        Assert.Equal(1, locality.Together(0, 1)!.Value, 5);
        Assert.Equal(0, locality.Step(cells, choice, rows, valid, Tags, scaling).item<float>(), 5);
    }

    [Fact]
    public void Prior_SeedsThePair_ByName()
    {
        var path = Path.GetTempFileName();
        try
        {
            File.WriteAllLines(path, ["glasses\tshoes\t0\t300", "glasses\tunknown_tag\t5\t10"]);
            using var locality = new LocalityNegatives(CPU, slotBits: 10) { MinObservations = 20, PriorCap = 50 };
            var loaded = locality.LoadPrior(path, new Dictionary<string, int> { ["glasses"] = 0, ["shoes"] = 1 });

            Assert.Equal(1, loaded);
            Assert.Equal(0, locality.Together(0, 1)!.Value, 5);
            Assert.Null(locality.Together(0, 2));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
