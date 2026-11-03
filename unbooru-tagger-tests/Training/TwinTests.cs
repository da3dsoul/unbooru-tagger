using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class TwinTests
{
    [Fact]
    public void SampleTwins_ArePositiveFirst_ThenImagesWithoutTheTag_MostOverlappingFirst()
    {
        // Tag 0 is the anchor (gloves), tag 1 a character, 2-11 everything else.
        var images = new List<IReadOnlyList<int>>
        {
            new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 },       // the positive
            new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 },          // a near-perfect twin
            new[] { 1, 2, 3, 4, 5, 6, 7, 10, 11 },        // a weaker one
            new[] { 2, 3, 4, 5, 6, 10, 11, 12, 13 },      // weaker still
            new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },   // carries the tag: never a twin
        };
        var statistics = TagStatistics.Build(images, 14, samplesPerTag: 1000);
        var sampler = new TagCentricBatchSampler(statistics, 2, 2, random: new Random(1))
        {
            TwinEvery = 1, TwinTags = [0], CharacterTags = Enumerable.Range(0, 14).Select(t => t == 1).ToArray(),
        };

        SampledImage[]? twins = null;
        for (var attempt = 0; attempt < 20 && twins?[0].Image != 0; attempt++)
            twins = sampler.SampleTwins(3);

        Assert.NotNull(twins);
        Assert.Equal(new[] { 0, 1, 2, 3 }, twins!.Select(t => t.Image));
        Assert.All(twins, t => Assert.True(t.IsTwin && !t.IsHardNegative && t.AnchorTag == 0));
    }

    [Fact]
    public void NoWindows_AreTheMostTwinLikeHalf_LeavingOutTheTagsPeak()
    {
        // One positive (row 0) and three twins, four windows each. The positive's windows 0 and 1 match every twin's window 0
        // exactly; window 2 matches none; window 3 matches too but holds the tag's peak.
        var shared = new[] { 1f, 0f, 0f, 0f };
        var cells = zeros(4, 4, 4);
        for (var b = 0; b < 4; b++)
            for (var k = 0; k < 4; k++)
                cells[b, .., k] = tensor(b == 0 && k == 2 ? new[] { 0f, 0f, 0f, 1f } : b > 0 && k > 0 ? new[] { 0f, 1f, 0f, 0f } : shared);
        var valid = ones(new long[] { 4, 4 }, ScalarType.Bool);
        using var choice = new GlimpseChoice(full(4, 4, 1L), tensor(new long[] { 0, 100, 200, 300 }).repeat(4, 1), zeros(4, 4, ScalarType.Int64), valid.clone());

        using var no = TwinWindows.NoWindows(cells, valid, 0, tensor(new long[] { 1, 2, 3 }), ones(3, ScalarType.Bool), choice, tensor(310L), tensor(10L));

        Assert.Equal(new[] { true, true, false, false }, no.data<bool>().ToArray());
    }
}

public class TwinPasteTests
{
    [Fact]
    public void Apply_PastesThePeakRegionIntoOneTwin_AControlFromElsewhereIntoAnother_AndNothingWithoutAConfidentPeak()
    {
        // Three 512px images; the positive (row 0) is 1 at its peak region and 0.5 elsewhere, the twins 0.
        var pixels = zeros(new long[] { 3, 3, 512, 512 });
        pixels[0].fill_(0.5f);
        pixels[0, .., TensorIndex.Slice(96, 160), TensorIndex.Slice(96, 160)].fill_(1f);
        var content = Enumerable.Repeat(new UnbooruTagger.Core.Encoding.LetterboxBox(0, 0, 512, 512), 3).ToArray();
        // A 16x16 map (stride 32) peaking at cell (4, 4): canvas (144, 144).
        var map = zeros(new long[] { 1, 2, 16, 16 });
        map[0, 0, 4, 4] = tensor(5f);
        var mask = ones(new long[] { 1, 1, 16, 16 });
        var scaling = new UnbooruTagger.Core.Scoring.LogitScaling(1f, 0f);

        var result = TwinPaste.Apply(pixels, content, map, mask, tensor(new[] { 1f, 0f }), scaling, 0, [1, 2], 1, 32, new Random(0));

        Assert.Equal(new[] { 1 }, result.Pasted);
        Assert.Equal(new[] { 2 }, result.Controls);
        Assert.Equal(1f, pixels[1, 0, 144, 144].item<float>(), 3);   // the peak's patch, at the same place
        Assert.Equal(0.5f, pixels[2].max().item<float>(), 3);         // the control: from elsewhere in the positive

        var weak = TwinPaste.Apply(zeros(new long[] { 3, 3, 512, 512 }), content, map, mask, tensor(new[] { 1f, 0f }), new UnbooruTagger.Core.Scoring.LogitScaling(1f, -10f), 0, [1, 2], 1, 32, new Random(0));
        Assert.True(weak.Gated);
        Assert.Empty(weak.Pasted);
    }
}
