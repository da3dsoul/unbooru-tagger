using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Tests.Core;

public class TagLocalizabilityTests
{
    [Theory]
    [InlineData("artist:someone")]
    [InlineData("series:touhou")]
    [InlineData("white_background")]
    [InlineData("gradient background")]
    [InlineData("photo_(medium)")]
    [InlineData("1990s_(style)")]
    [InlineData("from_behind")]
    [InlineData("solo")]
    [InlineData("Looking_At_Viewer")]
    [InlineData("rain")]
    [InlineData("blue_theme")]
    [InlineData("limited_palette")]
    public void Defaults_ExcludeWholeImageTags(string tag) => Assert.False(TagLocalizability.Default.IsLocalizable(tag));

    [Theory]
    [InlineData("eyeshadow")]          // "shadow" must not be a substring rule
    [InlineData("covered_navel")]      // nor "cover"
    [InlineData("standing_split")]     // nor "split"
    [InlineData("paint_palette")]      // an object, unlike "limited palette"
    [InlineData("signature")]          // explicit exception: a small real region
    [InlineData("character:hatsune_miku")]
    [InlineData("necktie")]
    [InlineData("rain_coat")]
    public void Defaults_KeepRealRegions(string tag) => Assert.True(TagLocalizability.Default.IsLocalizable(tag));

    [Fact]
    public void Exceptions_OverrideBroaderRules()
    {
        var rules = TagLocalizability.Parse("* background\n!city background\n# comment\n");
        Assert.False(rules.IsLocalizable("forest_background"));
        Assert.True(rules.IsLocalizable("city_background"));
        Assert.True(rules.IsLocalizable("hat"));
    }
}
