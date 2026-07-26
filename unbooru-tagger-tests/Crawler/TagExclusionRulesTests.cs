using UnbooruTagger.Crawler;

namespace UnbooruTagger.Tests.Crawler;

public class TagExclusionRulesTests
{
    private static readonly TagExclusionRules Empty = new(new HashSet<string>(), new HashSet<string>());

    [Theory]
    [InlineData("white_hair")]
    [InlineData("artist:someone")]
    [InlineData("series:sousou_no_frieren")]
    [InlineData("character:frieren")]
    public void IsExcluded_NeverTouchesNonMetaTags(string identity)
    {
        Assert.False(Empty.IsExcluded(identity));
    }

    [Fact]
    public void IsExcluded_TrueForMetaTagsByDefault()
    {
        Assert.True(Empty.IsExcluded("meta:bad_pixiv_id"));
        Assert.True(Empty.IsExcluded("meta:translation_request"));
    }

    [Theory]
    [InlineData("meta:pen_(medium)")]
    [InlineData("meta:oil_painting_(medium)")]
    [InlineData("meta:photoshop_(medium)")]
    public void IsExcluded_AutomaticallyCarvesOutMediumTags(string identity)
    {
        Assert.False(Empty.IsExcluded(identity));
    }

    [Fact]
    public void IsExcluded_ExplicitIncludeRescuesAMetaTagNotCoveredByTheMediumSuffix()
    {
        var rules = new TagExclusionRules(new HashSet<string>(), new HashSet<string> { "meta:scan" });

        Assert.False(rules.IsExcluded("meta:scan"));
    }

    [Fact]
    public void IsExcluded_ExplicitExcludeDropsATagOutsideMeta()
    {
        var rules = new TagExclusionRules(new HashSet<string> { "some_junk_general_tag" }, new HashSet<string>());

        Assert.True(rules.IsExcluded("some_junk_general_tag"));
    }

    [Fact]
    public void IsExcluded_ExplicitIncludeOutranksAnExplicitExclude()
    {
        var rules = new TagExclusionRules(
            new HashSet<string> { "meta:scan" },
            new HashSet<string> { "meta:scan" });

        Assert.False(rules.IsExcluded("meta:scan"));
    }
}
