using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class SpatialMaskTests
{
    [Fact]
    public void Build_MarksOnlyLocationsWhoseCenterFallsInsideTheContentBox()
    {
        // A 4x4 canvas letterboxed to a 4-wide x 2-tall content box centered vertically
        // (padding bars top and bottom) -- at a 1:1 grid-to-canvas stride, only the
        // middle two rows of a 4x4 grid should be valid.
        var boxes = new[] { new LetterboxBox(0, 1, 4, 2) };

        var mask = SpatialMask.Build(boxes, inputSize: 4, spatialHeight: 4, spatialWidth: 4);
        var values = mask.data<float>().ToArray();

        Assert.Equal(0f, values[0 * 4 + 0]); // row 0: padding
        Assert.Equal(1f, values[1 * 4 + 0]); // row 1: content
        Assert.Equal(1f, values[2 * 4 + 0]); // row 2: content
        Assert.Equal(0f, values[3 * 4 + 0]); // row 3: padding
    }

    [Fact]
    public void Build_FallsBackToOneValidCell_WhenContentIsThinnerThanAGridStride()
    {
        // An extreme aspect ratio can letterbox content thinner than one grid cell's
        // stride, so no cell center falls inside it -- must still mark exactly one
        // valid cell instead of leaving the whole mask empty (which would NaN the
        // localization loss's log-sum-exp).
        var boxes = new[] { new LetterboxBox(0, 15, 32, 1) };

        var mask = SpatialMask.Build(boxes, inputSize: 32, spatialHeight: 2, spatialWidth: 2);
        var validCount = mask.sum().item<float>();

        Assert.Equal(1f, validCount);
    }

    [Fact]
    public void Build_ProducesIndependentMasksPerImageInABatch()
    {
        var boxes = new[]
        {
            new LetterboxBox(0, 0, 4, 4), // full canvas: every cell valid
            new LetterboxBox(0, 2, 4, 2), // bottom half only
        };

        var mask = SpatialMask.Build(boxes, inputSize: 4, spatialHeight: 4, spatialWidth: 4);

        Assert.Equal(16f, mask[0].sum().item<float>());
        Assert.Equal(8f, mask[1].sum().item<float>());
    }
}
