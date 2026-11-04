using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class SelfMapWindowsTests
{
    [Fact]
    public void Windows_AreCentredOnThePeak_AndNoWindowsComeFromTheLowestResponse()
    {
        // A 256px canvas, stride 4 (a 64x64 map), D = 2. The tag (1, 0) responds 1 in the top-left quarter and 10 at one cell
        // (cell 40,40 = canvas 162,162); the bottom half responds -5. Content: the whole canvas.
        const int size = 256, stride = 4, side = size / stride;
        var map = zeros(1, 2, side, side);
        map[0, 0, TensorIndex.Slice(0, side / 2), TensorIndex.Slice(0, side / 2)] = tensor(1f);
        map[0, 0, TensorIndex.Slice(side / 2, side), TensorIndex.Slice()] = tensor(-5f);
        map[0, 0, 40, 40] = tensor(10f);
        var mask = ones(1, 1, side, side);
        var tags = tensor(new[] { 1f, 0f }, [1, 1, 2]);
        var valid = tensor(new[] { true }, [1, 1]);
        var content = tensor(new long[] { 0, 0, size, size }, [1, 4]);

        var (choice, labels) = SelfMapWindows.Windows(map, mask, tags, valid, content, size, stride, new Generator(seed: 1));
        var windows = choice.ToWindows()[0];
        var label = labels.data<float>().ToArray();

        Assert.Equal(SelfMapWindows.PerPositive, windows.Length);
        Assert.Equal(new[] { 1f, 1f, 0f, 0f }, label);
        // The peak cell's centre is (162, 162): the yes windows are centred there.
        Assert.Equal((64, 130, 130), (windows[0].Side, windows[0].X, windows[0].Y));
        Assert.Equal((128, 98, 98), (windows[1].Side, windows[1].X, windows[1].Y));
        // The no windows lie in the bottom half (the -5 region), clear of the peak.
        Assert.All(windows.Skip(2), w => Assert.True(w.Side == 64 && w.Y >= size / 2, $"{w}"));
    }

    [Fact]
    public void Windows_PaddedPositivesAreNotReal()
    {
        const int size = 256, stride = 4, side = size / stride;
        var map = randn(1, 2, side, side);
        var (choice, _) = SelfMapWindows.Windows(map, ones(1, 1, side, side), randn(1, 2, 2), tensor(new[] { true, false }, [1, 2]),
            tensor(new long[] { 0, 0, size, size }, [1, 4]), size, stride);
        var real = choice.Valid.data<bool>().ToArray();
        Assert.All(real.Take(SelfMapWindows.PerPositive), Assert.True);
        Assert.All(real.Skip(SelfMapWindows.PerPositive), Assert.False);
    }

    [Fact]
    public void Term_PushesYesWindowsUp_AndNoWindowsDown()
    {
        var scaling = new LogitScaleBias(initialScale: 1f, initialBias: 0f);
        var cells = zeros(1, 2, SelfMapWindows.PerPositive).requires_grad_();
        var tags = tensor(new[] { 1f, 0f }, [1, 1, 2]);
        var labels = tensor(new[] { 1f, 1f, 0f, 0f }, [1, 4]);
        SelfMapWindows.Term(cells, tags, labels, ones(1, 4, ScalarType.Bool), scaling).backward();
        var grad = cells.grad!.data<float>().ToArray(); // [D=2, K=4], row-major: dimension 0 first
        Assert.True(grad[0] < 0 && grad[1] < 0, "yes windows: descending the loss raises their score");
        Assert.True(grad[2] > 0 && grad[3] > 0, "no windows: descending the loss lowers their score");
        Assert.All(grad.Skip(4), g => Assert.Equal(0f, g));
    }

    [Fact]
    public void PeakWindowMask_KeepsOnlyEachSlotsPeakWindows()
    {
        var mask = SelfMapWindows.PeakWindowMask(2, CPU).data<bool>().ToArray();
        Assert.Equal(new[] { true, true, false, false, true, true, false, false }, mask);
    }
}
