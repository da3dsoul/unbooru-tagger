using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

public class TagCategoryTests
{
    [Fact]
    public void Identity_LeavesGeneralTagsUnprefixed()
    {
        Assert.Equal("white_hair", TagCategoryNaming.Identity("white_hair", TagCategory.General));
    }

    [Theory]
    [InlineData(TagCategory.Artist, "artist:frieren")]
    [InlineData(TagCategory.Copyright, "series:frieren")]
    [InlineData(TagCategory.Character, "character:frieren")]
    [InlineData(TagCategory.Meta, "meta:frieren")]
    public void Identity_PrefixesNonGeneralCategories(TagCategory category, string expected)
    {
        Assert.Equal(expected, TagCategoryNaming.Identity("frieren", category));
    }

    [Theory]
    [InlineData("white_hair", "white_hair")]
    [InlineData("artist:someone", "someone")]
    [InlineData("series:sousou_no_frieren", "sousou_no_frieren")]
    [InlineData("character:frieren", "frieren")]
    [InlineData("meta:highres", "highres")]
    public void RawName_IsTheInverseOfIdentity(string identity, string expectedRaw)
    {
        Assert.Equal(expectedRaw, TagCategoryNaming.RawName(identity));
    }

    [Theory]
    [InlineData("white_hair", "white_hair", TagCategory.General)]
    [InlineData("artist:someone", "someone", TagCategory.Artist)]
    [InlineData("series:sousou_no_frieren", "sousou_no_frieren", TagCategory.Copyright)]
    [InlineData("character:frieren", "frieren", TagCategory.Character)]
    [InlineData("meta:highres", "highres", TagCategory.Meta)]
    public void Split_RecoversBothRawNameAndCategory(string identity, string expectedRaw, TagCategory expectedCategory)
    {
        var (rawName, category) = TagCategoryNaming.Split(identity);
        Assert.Equal(expectedRaw, rawName);
        Assert.Equal(expectedCategory, category);
    }

    [Theory]
    [InlineData(0, TagCategory.General)]
    [InlineData(1, TagCategory.Artist)]
    [InlineData(3, TagCategory.Copyright)]
    [InlineData(4, TagCategory.Character)]
    [InlineData(5, TagCategory.Meta)]
    [InlineData(2, TagCategory.General)] // unused by either site
    [InlineData(6, TagCategory.General)] // Gelbooru's deprecated=6, no Danbooru equivalent
    [InlineData(999, TagCategory.General)] // unrecognized — falls back rather than guessing
    public void FromRawCode_MapsKnownCodesAndFallsBackToGeneral(int code, TagCategory expected)
    {
        Assert.Equal(expected, TagCategoryNaming.FromRawCode(code));
    }
}
