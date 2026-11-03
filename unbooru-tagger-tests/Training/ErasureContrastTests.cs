using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class ErasureContrastTests
{
    private const int Grid = 16;
    private const int Cell = 8;
    private static readonly int[] Content = [0, 0, Grid * Cell, Grid * Cell];
    private static bool[] AllValid() => Enumerable.Repeat(true, Grid * Grid).ToArray();

    [Fact]
    public void Region_IsThePeaksConnectedBlob_GrownByACell()
    {
        var heat = new float[Grid * Grid];
        heat[2 * Grid + 2] = 10;
        heat[2 * Grid + 3] = 9;
        heat[6 * Grid + 6] = 9; // as hot, but not connected to the peak
        var region = ErasureContrast.Region(heat, AllValid(), Grid, Cell, Content);
        Assert.Equal([1 * Cell, 1 * Cell, 4 * Cell, 3 * Cell], region!);
    }

    [Fact]
    public void Region_OverTheCap_IsASquareAroundThePeak()
    {
        var heat = Enumerable.Range(0, Grid * Grid).Select(i => i == 3 * Grid + 4 ? 10f : 5f).ToArray();
        heat[0] = 0; // the median sits at 5, so every cell but one is in the blob
        var region = ErasureContrast.Region(heat, AllValid(), Grid, Cell, Content)!;
        Assert.True(region[2] * region[3] <= ErasureContrast.MaxAreaFraction * Content[2] * Content[3]);
        Assert.InRange(4 * Cell + Cell / 2, region[0], region[0] + region[2]);
        Assert.InRange(3 * Cell + Cell / 2, region[1], region[1] + region[3]);
    }

    [Fact]
    public void Region_IgnoresPaddingCells()
    {
        var heat = new float[Grid * Grid];
        heat[0] = 100; // padding
        heat[5 * Grid + 5] = 10;
        var valid = AllValid();
        valid[0] = false;
        var region = ErasureContrast.Region(heat, valid, Grid, Cell, Content)!;
        Assert.Equal(4 * Cell, region[0]);
    }

    [Fact]
    public void Candidates_AreCalledLocalizablePositives_OnePerImage()
    {
        // 2 images x 3 tags; tag 2 isn't localizable. Image 0: tag 0 called, tag 1 not (the control's other tag). Image 1: only tag 2.
        using var targets = new PairTargets(
            tensor(new float[] { 1, 1, -1, -1, -1, 1 }, [2, 3]), ones(2, 3), zeros(2, 3), 0);
        float[] logits = [2, -2, 0, 0, 0, 5];
        var picked = ErasureContrast.Candidates(targets, logits, 1, 0, [true, true, false], 4, new Random(1));
        Assert.Equal([(0, 0, 1)], picked);
    }

    [Fact]
    public void Variants_BlankTheObjectThenTheControl()
    {
        using var pixels = ones(2, 3, 16, 16);
        var pairs = new List<ErasurePair> { new(1, 0, [0, 0, 4, 4], [8, 8, 4, 4]) };
        using var variants = ErasureContrast.Variants(pixels, pairs, 16, [OcclusionFill.Zero]);
        Assert.Equal([2L, 3, 16, 16], variants.shape);
        Assert.Equal(0f, variants[0, 0, 1, 1].item<float>());
        Assert.Equal(1f, variants[0, 0, 9, 9].item<float>());
        Assert.Equal(1f, variants[1, 0, 1, 1].item<float>());
        Assert.Equal(0f, variants[1, 0, 9, 9].item<float>());
    }

    [Fact]
    public void Hinge_IsZeroOnceTheObjectDropsAMarginMore_AndPositiveOtherwise()
    {
        using var scaling = new LogitScaleBias(1f, 0f);
        using var tags = tensor(new float[] { 1, 0 }, [1, 2]);
        var pairs = new List<ErasurePair> { new(0, 0, [0, 0, 1, 1], [1, 1, 1, 1]) };

        using var good = tensor(new float[] { -1, 0, 1, 0 }, [2, 2]); // object blanked -1, control blanked 1
        Assert.Equal(0f, ErasureContrast.Compute(good, tags, pairs, scaling, ErasureLoss.Hinge, 1f).item<float>(), 5);

        using var bad = tensor(new float[] { 1, 0, 1, 0 }, [2, 2]);  // blanking the object changed nothing
        Assert.Equal(1f, ErasureContrast.Compute(bad, tags, pairs, scaling, ErasureLoss.Hinge, 1f).item<float>(), 5);

        var negative = ErasureContrast.Compute(bad, tags, pairs, scaling, ErasureLoss.Negative, 1f).item<float>();
        Assert.Equal((float)(Math.Log(1 + Math.E) + Math.Log(1 + 1 / Math.E)), negative, 4);

        // Negative-only: just the object-blanked view as a negative; the control-blanked score doesn't enter unless anchored.
        var negativeOnly = ErasureContrast.Compute(bad, tags, pairs, scaling, ErasureLoss.NegativeOnly, 1f).item<float>();
        Assert.Equal((float)Math.Log(1 + Math.E), negativeOnly, 4);
        using var controlHigh = tensor(new float[] { 1, 0, 5, 0 }, [2, 2]);
        Assert.Equal(negativeOnly, ErasureContrast.Compute(controlHigh, tags, pairs, scaling, ErasureLoss.NegativeOnly, 1f).item<float>(), 4);
        Assert.Equal(negativeOnly + 0.5f * 2f, ErasureContrast.Compute(controlHigh, tags, pairs, scaling, ErasureLoss.NegativeOnly, 1f, [3f], 0.5f).item<float>(), 4);
    }

    [Fact]
    public void Anchor_AddsHowFarTheControlMovedFromTheUnblankedScore()
    {
        using var scaling = new LogitScaleBias(1f, 0f);
        using var tags = tensor(new float[] { 1, 0 }, [1, 2]);
        var pairs = new List<ErasurePair> { new(0, 0, [0, 0, 1, 1], [1, 1, 1, 1]) };
        using var good = tensor(new float[] { -1, 0, 1, 0 }, [2, 2]); // hinge satisfied, control blanked 1

        Assert.Equal(1f, ErasureContrast.Compute(good, tags, pairs, scaling, ErasureLoss.Hinge, 1f, [3f], 0.5f).item<float>(), 5);
        Assert.Equal(0.25f, ErasureContrast.Compute(good, tags, pairs, scaling, ErasureLoss.Hinge, 1f, [0.5f], 0.5f).item<float>(), 5);
        Assert.Equal(0f, ErasureContrast.Compute(good, tags, pairs, scaling, ErasureLoss.Hinge, 1f, [3f], 0f).item<float>(), 5);
    }

    [Fact]
    public void ControlOnOther_IsTheRegionsSize_OnTheOtherTagsPeak_UnlessItOverlaps()
    {
        var heat = new float[Grid * Grid];
        heat[12 * Grid + 12] = 5;
        int[] region = [8, 8, 24, 16];
        var control = ErasureContrast.ControlOnOther(heat, AllValid(), Grid, Cell, Content, region)!;
        Assert.Equal([12 * Cell + Cell / 2 - 12, 12 * Cell + Cell / 2 - 8, 24, 16], control);

        heat[12 * Grid + 12] = 0;
        heat[2 * Grid + 2] = 5; // the other tag peaks inside the region
        Assert.Null(ErasureContrast.ControlOnOther(heat, AllValid(), Grid, Cell, Content, region));

        heat[2 * Grid + 2] = 0;
        heat[Grid * Grid - 1] = 5; // at the corner: shifted inside the content
        Assert.Equal([Grid * Cell - 24, Grid * Cell - 16, 24, 16], ErasureContrast.ControlOnOther(heat, AllValid(), Grid, Cell, Content, region));
    }

    [Fact]
    public void Fills_MeanUsesTheImagesColor_AndOnlyTheRectangleChanges()
    {
        var image = new float[3 * 4 * 4];
        for (var i = 0; i < image.Length; i++)
            image[i] = i / 16; // channel c is all c
        image[0] = 4; // channel 0 mean 0.25
        OcclusionFills.Apply(image, [new[] { 2, 2, 2, 2 }], 4, OcclusionFill.Mean, new Random(1));
        Assert.Equal(0.25f, image[2 * 4 + 2]);
        Assert.Equal(1f, image[16 + 3 * 4 + 3]);
        Assert.Equal(0f, image[1]);

        using var pixels = tensor(new float[] { 1, 3, 1, 3, 2, 2, 2, 2, 0, 0, 0, 0 }, [1, 3, 2, 2]);
        using var filled = OcclusionFills.Images(pixels, [OcclusionFill.Mean]);
        Assert.Equal(2f, filled[0, 0, 1, 1].item<float>());
    }
}
