using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Tests.Training;

public class CharacterExtentTests
{
    // Vocabulary: 0 solo, 1 character a, 2 character b, 3 dress, 4 background (whole-image).
    private static readonly bool[] Localizable = [false, true, true, true, false];
    private static readonly bool[] IsCharacter = [false, true, true, false, false];

    private static PairTargets Targets(params float[][] rows)
    {
        var flat = rows.SelectMany(r => r).ToArray();
        var shape = new long[] { rows.Length, rows[0].Length };
        return new PairTargets(tensor(flat, shape), ones(shape), zeros(shape), flat.Sum());
    }

    [Fact]
    public void Select_TakesSoloImagesWithOneCharacter_AndTheirOtherLocalizableTags()
    {
        using var targets = Targets(
            [1, 1, 0, 1, 1], // solo, a, dress, background: picked, figure from dress only
            [1, 1, 1, 1, 0], // solo but two characters
            [0, 1, 0, 1, 0], // not solo
            [1, 0, 0, 1, 0]); // solo, no character

        var picked = Assert.Single(CharacterExtent.Select(targets, Localizable, IsCharacter, soloRow: 0, max: 0, needOthers: true, new Random(1)));

        Assert.Equal((0, 1), (picked.Image, picked.Character));
        Assert.Equal([3], picked.Others);
    }

    [Fact]
    public void ExtentTerm_PullsTheCharactersMapOntoTheFigure_AndOffTheRest()
    {
        // 2 channels on a 4x4 map: channel 0 is the character's, channel 1 the dress's. The character fires on one cell (the face);
        // the dress covers the top half.
        var features = zeros(1, 2, 4, 4);
        features[0, 0, 0, 0] = tensor(2f);
        features[0, 1, 0] = tensor(1f);
        features[0, 1, 1] = tensor(1f);
        var map = new TorchSharp.Modules.Parameter(features.clone());
        var embeddings = tensor(new float[] { 0, 0, 1, 0, 0, 0, 0, 1, 0, 0 }, [5, 2]);
        using var scaling = new LogitScaleBias(initialScale: 4, initialBias: 0);
        using var mask = ones(1, 1, 4, 4);
        var images = new List<CharacterExtentImage> { new(0, 1, [3]) };

        var optimizer = optim.SGD([map], 0.5);
        var before = CharacterExtent.ExtentTerm(map, mask, embeddings, scaling, images).item<float>();
        for (var i = 0; i < 50; i++)
        {
            optimizer.zero_grad();
            CharacterExtent.ExtentTerm(map, mask, embeddings, scaling, images).backward();
            optimizer.step();
        }
        var after = CharacterExtent.ExtentTerm(map, mask, embeddings, scaling, images).item<float>();
        var character = map[0, 0].detach();

        Assert.True(after < before / 2, $"{before} -> {after}");
        // A figure cell far from the face is now within the band below the character's peak (in scaled log-odds).
        var peak = character.max().item<float>();
        Assert.True((peak - character[1, 3].item<float>()) * 4 <= CharacterExtent.ExtentBand + 0.5f);
    }

    [Fact]
    public void HideTerm_IsLowWhenTheHiddenViewStillFindsTheCharacter()
    {
        var embeddings = tensor(new float[] { 0, 0, 1, 0, 0, 0, 0, 1, 0, 0 }, [5, 2]);
        using var scaling = new LogitScaleBias(initialScale: 4, initialBias: -2);
        using var mask = ones(2, 1, 2, 2);
        var found = zeros(1, 2, 2, 2);
        found[0, 0, 1, 1] = tensor(3f);
        var lost = zeros(1, 2, 2, 2);
        var images = new List<CharacterExtentImage> { new(0, 1, []) };

        var low = CharacterExtent.HideTerm(found, mask.narrow(0, 0, 1), embeddings, scaling, images, temperature: 0.25f).item<float>();
        var high = CharacterExtent.HideTerm(lost, mask.narrow(0, 0, 1), embeddings, scaling, images, temperature: 0.25f).item<float>();

        Assert.True(low < 0.1f && high > 1f, $"{low} vs {high}");
    }
}
