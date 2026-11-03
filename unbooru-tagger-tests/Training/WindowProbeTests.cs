using UnbooruTagger.Training.Commands;
using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using TorchSharp;

namespace UnbooruTagger.Tests.Training;

public class WindowProbeTests
{
    [Fact]
    public void Label_IsYesWhenTheWindowHoldsOrIsFilledByABox_NoWhenItTouchesNone_ElseIgnored()
    {
        var window = new GlimpseWindow(1, 100, 100, 64);
        int[][] none = [];

        Assert.Equal(1f, WindowProbeCommandHandler.Label(window, [[110, 110, 20, 20]], none));   // holds a small box
        Assert.Equal(1f, WindowProbeCommandHandler.Label(window, [[60, 60, 200, 200]], none));   // filled by a large one
        Assert.Equal(1f, WindowProbeCommandHandler.Label(window, [[150, 110, 20, 20]], none));   // half of the box inside
        Assert.Equal(-1f, WindowProbeCommandHandler.Label(window, [[160, 110, 20, 20]], none));  // a sliver: neither
        Assert.Equal(0f, WindowProbeCommandHandler.Label(window, [[300, 300, 20, 20]], none));   // elsewhere
        Assert.Equal(-1f, WindowProbeCommandHandler.Label(window, [[300, 300, 20, 20]], [[90, 90, 20, 20]])); // touches an ignore box
    }
}

public class MosaicGlimpseTests
{
    [Fact]
    public void GlimpseTerm_SaysNoOnOtherTiles_YesInsideItsOwnTile_AndNothingWhereAWindowStraddlesBoth()
    {
        // Four images, one canvas; tag 1 is on image 0 only, a labelled "no" on images 1-3. Tag 0 isn't localizable.
        using var labels = torch.tensor(new float[] { 0, 1, 0, -1, 0, -1, 0, -1 }, [4, 2]);
        using var weights = torch.ones(4, 2);
        using var focal = torch.zeros(4, 2);
        var targets = new PairTargets(labels, weights, focal, 1f, null);
        var (rows, valid, tileLabels) = Mosaic.GlimpseTileLabels(targets, [0, 1, 2, 3], [false, true], 2, new Random(0));
        Assert.Equal(new long[] { 1, 0 }, rows);
        Assert.Equal(new[] { true, false }, valid);
        Assert.Equal(new float[] { 1, -1, 0, -1, 0, -1, 0, -1 }, tileLabels);

        // Windows (zoom 1 = 64px): inside tile 0, inside tile 1, straddling tiles 0 and 1.
        using var choice = new GlimpseChoice(torch.tensor(new long[] { 1, 1, 1 }, [1, 3]), torch.tensor(new long[] { 10, 300, 230 }, [1, 3]),
            torch.tensor(new long[] { 10, 10, 10 }, [1, 3]), torch.ones(new long[] { 1, 3 }, torch.ScalarType.Bool));
        var (overlaps, inside) = Mosaic.WindowTiles(choice, 256);
        using var cells = torch.randn(1, 4, 3).requires_grad_();
        using var tags = torch.randn(1, 2, 4);
        using var scaling = new LogitScaleBias(1f, 0f);
        var term = Mosaic.GlimpseTerm(cells, tags, torch.tensor(valid, [1, 2]), torch.tensor(tileLabels, [1, 4, 2]), overlaps, inside, 0.5f, scaling);
        term.backward();

        var grad = cells.grad!.abs().sum(1).squeeze(0).data<float>().ToArray();
        Assert.True(grad[0] > 0); // the bag
        Assert.True(grad[1] > 0); // the "no"
        Assert.Equal(0f, grad[2]); // touches the tile that carries it
    }
}
