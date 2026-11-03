using UnbooruTagger.Core.Encoding;
using UnbooruTagger.Training.Model;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class GlimpsesTests
{
    [Fact]
    public void ObjectOnly_KeepsTheObjectHalf_AndHoldsTheContextHalfAtItsMean()
    {
        using var cells = randn(2, 6, 5); // [B, D, K]
        var mean = new[] { 1f, 2f, 3f };

        using var split = Glimpses.ObjectOnly(cells, 3, mean);

        Assert.True(split.narrow(1, 0, 3).allclose(cells.narrow(1, 0, 3)));
        Assert.Equal(new[] { 1f, 2f, 3f }, split[1, TensorIndex.Slice(3, 6), 4].data<float>().ToArray());
        // Without a mean, zeros: a tag's score shifts by a constant, which every pooling keeps.
        using var zeroed = Glimpses.ObjectOnly(cells, 3, (float[]?)null);
        Assert.Equal(0f, zeroed.narrow(1, 3, 3).abs().sum().item<float>());
    }

    [Fact]
    public void Layout_At512_Has256_64_49_9WindowsOf32_64_128_256Pixels()
    {
        var layout = Glimpses.Layout(512);

        Assert.Equal(378, layout.Length);
        Assert.Equal([256, 64, 49, 9], Enumerable.Range(0, 4).Select(z => layout.Count(w => w.Zoom == z)));
        Assert.Equal([32, 64, 128, 256], Enumerable.Range(0, 4).Select(z => layout.First(w => w.Zoom == z).Side));
        // Zoomed out, windows overlap by half; the last one ends at the edge.
        Assert.Contains(layout, w => w.Zoom == 3 && w.X == 128 && w.Y == 256);
        Assert.All(layout, w => Assert.True(w.X + w.Side <= 512 && w.Y + w.Side <= 512));
    }

    [Fact]
    public void Windows_MostlyOnTheLetterbox_AreLeftOut()
    {
        var layout = Glimpses.Layout(512);
        var wide = new LetterboxBox(0, 128, 512, 256); // bars above and below

        var all = Glimpses.All(layout, [wide])[0];

        Assert.All(all, g => Assert.True(Glimpses.ContentFraction(layout[g], wide) >= Glimpses.MinContent));
        Assert.DoesNotContain(all, g => layout[g].Y + layout[g].Side <= 128);
        Assert.Contains(all, g => layout[g].Zoom == 3 && layout[g].Y == 128);
    }

    [Fact]
    public void AThinStrip_StillGetsEachZoomsFullestWindow()
    {
        var layout = Glimpses.Layout(512);
        var strip = new LetterboxBox(0, 250, 512, 12);

        var all = Glimpses.All(layout, [strip])[0];

        Assert.Equal(4, all.Length);
        Assert.Equal([0, 1, 2, 3], all.Select(g => layout[g].Zoom).Order());
        Assert.NotEmpty(Glimpses.Sample(layout, strip, [], 16, new Random(1)));
    }

    [Fact]
    public void Sample_FramesEachTargetPoint_ThenFillsWithRandomWindows()
    {
        var layout = Glimpses.Layout(512);
        var content = new LetterboxBox(0, 0, 512, 512);

        var chosen = Glimpses.Sample(layout, content, [(400, 40), (100, 450)], 16, new Random(1));

        Assert.Equal(16, chosen.Length);
        Assert.Equal(16, chosen.Distinct().Count());
        Assert.True(Contains(layout[chosen[0]], 400, 40));
        Assert.True(Contains(layout[chosen[1]], 100, 450));
    }

    [Fact]
    public void Embed_CropsAtEachZoom_AndAddsTheZoomCode()
    {
        // A tower whose embedding we can predict isn't practical; check the crops through a shape-and-code test instead:
        // two glimpses of one image, same window content at zoom 1, differ only by their zoom codes.
        var tower = new ImageTower(8, stemChannels: 8, stageChannels: [8, 16], blocksPerStage: [1, 1], fusedStages: 1, classificationGlimpses: true);
        using (no_grad())
            tower.GlimpseZoomCodes!.copy_(arange(0, 4 * 8, dtype: ScalarType.Float32).reshape(4, 8));
        var layout = Glimpses.Layout(128);
        using var pixels = zeros(2, 3, 128, 128); // flat images: every crop is the same, whatever its zoom
        var content = new[] { new LetterboxBox(0, 0, 128, 128), new LetterboxBox(0, 0, 128, 128) };
        var zoom1 = Array.FindIndex(layout, w => w.Zoom == 1);
        var zoom2 = Array.FindIndex(layout, w => w.Zoom == 0);

        var (cells, valid) = Glimpses.Embed(tower, pixels, content, layout, [[zoom1, zoom2], [zoom1]]);

        Assert.Equal([2L, 8, 2], cells.shape);
        Assert.Equal([true, true, true, false], valid.data<bool>().ToArray());
        var difference = (cells[0, .., 1] - cells[0, .., 0]).data<float>().ToArray();
        Assert.All(difference, d => Assert.Equal(-8f, d, 4)); // zoom index 0's code minus zoom index 1's
    }

    [Fact]
    public void Embed_Padded_GivesTheSameGlimpses_WithBlankSlotsInvalid()
    {
        var tower = new ImageTower(8, stemChannels: 8, stageChannels: [8, 16], blocksPerStage: [1, 1], fusedStages: 1, classificationGlimpses: true);
        var layout = Glimpses.Layout(128);
        manual_seed(3);
        using var pixels = randn(2, 3, 128, 128);
        var content = new[] { new LetterboxBox(0, 0, 128, 128), new LetterboxBox(0, 0, 128, 96) };
        int[][] windows = [[0, 5, 20], [3]];

        var (cells, valid) = Glimpses.Embed(tower, pixels, content, layout, windows);
        // Padded to 8 glimpses (K = 4 per image), in passes of 3: 9 run, the last 5 blank.
        var (padded, paddedValid) = Glimpses.Embed(tower, pixels, content, layout, windows, chunk: 3, padTo: 8);

        Assert.Equal([2L, 8, 4], padded.shape);
        Assert.Equal([true, true, true, false, true, false, false, false], paddedValid.data<bool>().ToArray());
        Assert.True(padded.narrow(2, 0, 3).allclose(cells, rtol: 1e-4, atol: 1e-5));
    }

    private static readonly LetterboxBox Full = new(0, 0, 512, 512);

    private static GlimpseWindow[] Chosen(Tensor proposals, Tensor map, Tensor tags, int count, LetterboxBox[]? content = null)
    {
        content ??= [Full];
        using var mapMask = SpatialMask.Build(content, 512, map.shape[2], map.shape[3]);
        using var boxes = Glimpses.ContentBoxes(content, CPU);
        using var choice = Glimpses.Propose(proposals, map, tags, mapMask, boxes, 512, outputStride: 4, count: count);
        return choice.ToWindows()[0];
    }

    [Fact]
    public void Propose_LooksAtTheTopProposedTagsPeak_WithTheSensor_WithoutLabels()
    {
        // A 512px canvas at stride 4: a 128x128 map with two channels. Tag 0 answers a 2x2-cell spot at canvas (402, 82),
        // tag 1 a blob lower left; the proposals rank tag 0 first.
        using var map = zeros(1, 2, 128, 128);
        map[0, 0, TensorIndex.Slice(20, 22), TensorIndex.Slice(100, 102)] = tensor(10f);
        map[0, 1, TensorIndex.Slice(70, 110), TensorIndex.Slice(10, 50)] = tensor(5f);
        using var tags = tensor(new float[] { 1, 0, 0, 1, -1, -1 }, [3, 2]);
        using var proposals = tensor(new float[] { 2, 1 }, [1, 2]);

        var chosen = Chosen(proposals, map, tags, count: 4);

        // The budget goes to the top tag: one window per zoom, centred on its peak.
        Assert.Equal(Glimpses.Sensor(402, 82, Full, 512), chosen);
        Assert.Equal([32, 64, 128, 256], chosen.Select(w => w.Side));
    }

    [Fact]
    public void Propose_DoesNotLookTwiceAtOneSpot_AndFillsOutWithTheZoomedOutGrid()
    {
        using var map = zeros(1, 1, 128, 128);
        map[0, 0, TensorIndex.Slice(60, 62), TensorIndex.Slice(60, 62)] = tensor(10f);
        using var tags = tensor(new float[] { 1, 2 }, [2, 1]); // two tags answering the same spot
        using var proposals = tensor(new float[] { 1 }, [1, 1]);

        var chosen = Chosen(proposals, map, tags, count: 6);

        Assert.Equal(6, chosen.Length);
        Assert.Single(chosen, w => w.Side == 32);
        Assert.Equal(3, chosen.Count(w => w.Side == 256)); // the sensor's, then two grid windows
    }

    [Fact]
    public void Propose_KeepsEveryWindowInsideTheContent()
    {
        manual_seed(5);
        using var map = randn(1, 4, 128, 128);
        using var tags = randn(6, 4);
        using var proposals = randn(1, 4);
        var wide = new LetterboxBox(0, 128, 512, 256);

        var chosen = Chosen(proposals, map, tags, count: 16, content: [wide]);

        Assert.Equal(16, chosen.Length);
        Assert.All(chosen, w => Assert.True(w.Y >= 128 && w.Y + w.Side <= 384 && w.X >= 0 && w.X + w.Side <= 512, $"{w}"));
    }

    [Fact]
    public void Propose_NeverChoosesTheSameLookTwice_AndCountsItsZooms()
    {
        // Smooth random maps, so neighbouring tags peak near each other and their zoomed-out windows overlap.
        manual_seed(9);
        using var map = nn.functional.avg_pool2d(randn(3, 8, 128, 128), 17, stride: 1, padding: 8);
        using var tags = randn(40, 8);
        using var proposals = randn(3, 8);
        var content = new[] { Full, Full, new LetterboxBox(0, 128, 512, 256) };
        using var mapMask = SpatialMask.Build(content, 512, 128, 128);
        using var boxes = Glimpses.ContentBoxes(content, CPU);

        using var choice = Glimpses.Propose(proposals, map, tags, mapMask, boxes, 512, outputStride: 4, count: 48);
        using var counted = Glimpses.ZoomCounts(choice);
        var windows = choice.ToWindows();
        Assert.Equal(Enumerable.Range(0, Glimpses.Zooms.Length).Select(z => (long)windows.Sum(w => w.Count(x => x.Zoom == z))), counted.data<long>().ToArray());

        foreach (var chosen in windows)
        {
            Assert.Equal(48, chosen.Length);
            for (var i = 0; i < chosen.Length; i++)
                for (var j = i + 1; j < chosen.Length; j++)
                    Assert.False(Glimpses.SameLookAs(chosen[i], chosen[j]), $"{chosen[i]} and {chosen[j]}");
        }
    }

    [Fact]
    public void Crops_AreTheWindowsOwnPixels_AtEachZoom()
    {
        // A patterned image (every pixel distinct), so a crop from the wrong place or image can't pass.
        using var pixels = arange(2 * 3 * 128 * 128, dtype: ScalarType.Float32).reshape(2, 3, 128, 128) / 1000f;
        GlimpseWindow[] windows = [new(1, 37, 11, 64), new(2, 0, 0, 128), new(0, 70, 90, 32), new(1, 5, 60, 64)];
        using var choice = new GlimpseChoice(
            tensor(windows.Select(w => (long)w.Zoom).ToArray(), [2, 2]), tensor(windows.Select(w => (long)w.X).ToArray(), [2, 2]),
            tensor(windows.Select(w => (long)w.Y).ToArray(), [2, 2]), ones(2, 2, dtype: ScalarType.Bool));

        using var crops = Glimpses.Crops(pixels, choice);

        Assert.Equal([4L, 3, Glimpses.Size, Glimpses.Size], crops.shape);
        for (var k = 0; k < windows.Length; k++)
        {
            var w = windows[k];
            using var region = pixels[k / 2, .., w.Y..(w.Y + w.Side), w.X..(w.X + w.Side)].unsqueeze(0);
            using var expected = w.Side == Glimpses.Size ? region.clone()
                : w.Side > Glimpses.Size ? nn.functional.avg_pool2d(region, w.Side / Glimpses.Size)
                : nn.functional.interpolate(region, size: [Glimpses.Size, Glimpses.Size], mode: InterpolationMode.Bilinear, align_corners: false);
            // Zooming in resamples the whole canvas, so a window's edge pixels blend with its neighbours: compare the interior.
            var inner = w.Side < Glimpses.Size ? 2 : 0;
            using var got = crops[k].unsqueeze(0)[.., .., inner..(Glimpses.Size - inner), inner..(Glimpses.Size - inner)];
            using var want = expected[.., .., inner..(Glimpses.Size - inner), inner..(Glimpses.Size - inner)];
            Assert.True(got.allclose(want, rtol: 1e-4, atol: 1e-3), $"window {k} (zoom {w.Zoom} at {w.X},{w.Y}): max error {(got - want).abs().max().item<float>()}");
        }
    }

    [Fact]
    public void ASeparateGlimpseTower_EmbedsTheGlimpses_AndStartsAsACopy()
    {
        manual_seed(13);
        var tower = new ImageTower(8, stemChannels: 8, stageChannels: [8, 16], blocksPerStage: [1, 1], fusedStages: 1, classificationGlimpses: true);
        using (no_grad())
            tower.GlimpseZoomCodes!.normal_();
        using var pixels = randn(1, 3, 128, 128);
        var content = new[] { new LetterboxBox(0, 0, 128, 128) };
        GlimpseWindow[] windows = [new(0, 10, 20, 32), new(1, 64, 0, 64), new(2, 0, 0, 128)];
        using var choice = new GlimpseChoice(
            tensor(windows.Select(w => (long)w.Zoom).ToArray(), [1, 3]), tensor(windows.Select(w => (long)w.X).ToArray(), [1, 3]),
            tensor(windows.Select(w => (long)w.Y).ToArray(), [1, 3]), ones(1, 3, dtype: ScalarType.Bool));
        using var boxes = Glimpses.ContentBoxes(content, CPU);
        var (shared, _) = Glimpses.Embed(tower, pixels, boxes, choice);

        tower.GlimpseTower = tower.CopyForGlimpses();
        var (copied, _) = Glimpses.Embed(tower, pixels, boxes, choice);
        Assert.True(copied.allclose(shared, rtol: 1e-5, atol: 1e-6));

        using (no_grad())
            tower.GlimpseTower.Projection.weight!.mul_(2f);
        var (changed, _) = Glimpses.Embed(tower, pixels, boxes, choice);
        Assert.False(changed.allclose(shared, rtol: 1e-3, atol: 1e-3));
        // The copy's weights are its own: the main tower's are untouched, and without the copy the glimpses are as before.
        Assert.False(tower.Projection.weight!.allclose(tower.GlimpseTower.Projection.weight!));
        tower.GlimpseTower = null;
        Assert.True(Glimpses.Embed(tower, pixels, boxes, choice).Cells.allclose(shared, rtol: 1e-5, atol: 1e-6));
    }

    [Fact]
    public void Embed_FromADeviceChoice_GivesTheLayoutEmbedsGlimpses()
    {
        manual_seed(11);
        var tower = new ImageTower(8, stemChannels: 8, stageChannels: [8, 16], blocksPerStage: [1, 1], fusedStages: 1, classificationGlimpses: true);
        using (no_grad())
            tower.GlimpseZoomCodes!.normal_();
        using var pixels = randn(2, 3, 128, 128);
        var content = new[] { new LetterboxBox(0, 0, 128, 128), new LetterboxBox(0, 16, 128, 96) };
        GlimpseWindow[][] windows =
        [
            [new(0, 10, 20, 32), new(1, 64, 0, 64), new(2, 0, 0, 128)],
            [new(1, 33, 40, 64), new(0, 95, 7, 32), new(2, 0, 0, 128)],
        ];
        var layout = windows.SelectMany(w => w).ToArray();
        using var choice = new GlimpseChoice(
            tensor(layout.Select(w => (long)w.Zoom).ToArray(), [2, 3]), tensor(layout.Select(w => (long)w.X).ToArray(), [2, 3]),
            tensor(layout.Select(w => (long)w.Y).ToArray(), [2, 3]), ones(2, 3, dtype: ScalarType.Bool));
        using var boxes = Glimpses.ContentBoxes(content, CPU);

        var (expected, _) = Glimpses.Embed(tower, pixels, content, layout, [[0, 1, 2], [3, 4, 5]]);
        var (cells, valid) = Glimpses.Embed(tower, pixels, boxes, choice);

        Assert.Equal(expected.shape, cells.shape);
        Assert.True(cells.allclose(expected, rtol: 1e-4, atol: 1e-5));
        Assert.All(valid.data<bool>().ToArray(), Assert.True);

        // With the 128px zoom left out of pooling (GlimpsePoolZooms 2 here, at 128px input), its windows are embedded the same
        // but not real for pooling.
        tower.GlimpsePoolZooms = 2;
        var (same, pooledValid) = Glimpses.Embed(tower, pixels, boxes, choice);
        Assert.True(same.allclose(cells));
        Assert.Equal([true, true, false, true, true, false], pooledValid.data<bool>().ToArray());
    }

    [Fact]
    public void Framed_OnTheDevice_IsFrames()
    {
        var random = new Random(3);
        var content = new[] { Full, new LetterboxBox(0, 128, 512, 256) };
        GlimpseWindow Random(int image)
        {
            var zoom = random.Next(Glimpses.Zooms.Length);
            return Glimpses.Centered(zoom, random.Next(512), random.Next(512), content[image], 512);
        }
        var chosen = Enumerable.Range(0, 2).Select(i => Enumerable.Range(0, 24).Select(_ => Random(i)).ToArray()).ToArray();
        var flat = chosen.SelectMany(c => c).ToArray();
        using var choice = new GlimpseChoice(
            tensor(flat.Select(w => (long)w.Zoom).ToArray(), [2, 24]), tensor(flat.Select(w => (long)w.X).ToArray(), [2, 24]),
            tensor(flat.Select(w => (long)w.Y).ToArray(), [2, 24]), ones(2, 24, dtype: ScalarType.Bool));
        // Peaks: half near a chosen window's centre, half anywhere.
        var peaks = Enumerable.Range(0, 2).Select(i => Enumerable.Range(0, 40).Select(k => k % 2 == 0
            ? (chosen[i][k % 24].X + chosen[i][k % 24].Side / 2 + random.Next(-8, 9), chosen[i][k % 24].Y + chosen[i][k % 24].Side / 2 + random.Next(-8, 9))
            : (random.Next(512), random.Next(512))).ToArray()).ToArray();
        using var x = tensor(peaks.SelectMany(p => p.Select(q => (long)q.Item1)).ToArray(), [2, 40]);
        using var y = tensor(peaks.SelectMany(p => p.Select(q => (long)q.Item2)).ToArray(), [2, 40]);
        using var boxes = Glimpses.ContentBoxes(content, CPU);

        var framed = Glimpses.Framed(choice, x, y, boxes, 512).data<bool>().ToArray();

        var expected = Enumerable.Range(0, 2).SelectMany(i => peaks[i].Select(p => Glimpses.Frames(chosen[i], Glimpses.Sensor(p.Item1, p.Item2, content[i], 512)))).ToArray();
        Assert.Equal(expected, framed);
        Assert.Contains(true, framed);
        Assert.Contains(false, framed);
    }

    [Fact]
    public void FramingWindows_AreTheChosenWindowsThatAreOneOfTheSensorsLooks()
    {
        var random = new Random(5);
        var content = new[] { Full, new LetterboxBox(0, 128, 512, 256) };
        var peaks = new[] { (200, 220), (300, 200) };
        // Each image: its peak's sensor windows (all zooms), then random windows.
        var chosen = Enumerable.Range(0, 2).Select(i => Glimpses.Sensor(peaks[i].Item1, peaks[i].Item2, content[i], 512)
            .Concat(Enumerable.Range(0, 12).Select(_ => Glimpses.Centered(random.Next(Glimpses.Zooms.Length), random.Next(512), random.Next(512), content[i], 512)))
            .ToArray()).ToArray();
        var flat = chosen.SelectMany(c => c).ToArray();
        var k = chosen[0].Length;
        using var choice = new GlimpseChoice(
            tensor(flat.Select(w => (long)w.Zoom).ToArray(), [2, k]), tensor(flat.Select(w => (long)w.X).ToArray(), [2, k]),
            tensor(flat.Select(w => (long)w.Y).ToArray(), [2, k]), ones(2, k, dtype: ScalarType.Bool));
        using var x = tensor(peaks.Select(p => (long)p.Item1).ToArray(), [2, 1]);
        using var y = tensor(peaks.Select(p => (long)p.Item2).ToArray(), [2, 1]);
        using var boxes = Glimpses.ContentBoxes(content, CPU);

        var framing = Glimpses.FramingWindows(choice, x, y, boxes, 512).data<bool>().ToArray(); // [2, 1, K]

        for (var i = 0; i < 2; i++)
        {
            var sensor = Glimpses.Sensor(peaks[i].Item1, peaks[i].Item2, content[i], 512);
            for (var w = 0; w < k; w++)
                Assert.Equal(sensor.Any(s => Glimpses.SameLookAs(s, chosen[i][w])), framing[i * k + w]);
            Assert.All(Enumerable.Range(0, Glimpses.Zooms.Length), w => Assert.True(framing[i * k + w])); // the sensor itself
        }
    }

    [Fact]
    public void Frames_NeedsACloseLook_NotJustTheZoomedOutWindow()
    {
        var content = new LetterboxBox(0, 0, 512, 512);
        var sensor = Glimpses.Sensor(100, 100, content, 512);

        Assert.False(Glimpses.Frames([sensor[3]], sensor));
        Assert.True(Glimpses.Frames([sensor[1]], sensor));
        Assert.True(Glimpses.Frames([sensor[3], new GlimpseWindow(2, 40, 40, 128)], sensor)); // close enough to the sensor's 128px window
    }

    [Fact]
    public void Centered_StaysInsideTheContent()
    {
        var wide = new LetterboxBox(0, 128, 512, 256);

        Assert.Equal(new GlimpseWindow(3, 0, 128, 256), Glimpses.Centered(3, 10, 130, wide, 512));
        Assert.Equal(new GlimpseWindow(0, 480, 352, 32), Glimpses.Centered(0, 511, 383, wide, 512));
    }

    private static bool Contains(GlimpseWindow w, int x, int y) => x >= w.X && x < w.X + w.Side && y >= w.Y && y < w.Y + w.Side;
}
