using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Tests.Core;

public class ExclusiveTagGroupsTests
{
    private static TagVocabulary Vocabulary(params string[] tags)
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        foreach (var tag in tags)
            vocabulary.AddTag(tag);
        return vocabulary;
    }

    [Fact]
    public void Parse_MatchesUnderscoreNamesAgainstSpacedVocabularyTags()
    {
        var vocabulary = Vocabulary("solo", "multiple girls");
        var groups = ExclusiveTagGroups.Parse("solo, multiple_girls", vocabulary);

        Assert.Equal([1], groups.ExcludedBy(0));
        Assert.Equal([0], groups.ExcludedBy(1));
    }

    [Fact]
    public void Defaults_NeverMakeSoloAndOneGirlExclusive()
    {
        // solo + 1girl is the single most common pairing on any booru.
        var vocabulary = Vocabulary("solo", "1girl", "1boy", "multiple girls", "2girls");
        var groups = ExclusiveTagGroups.Parse(ExclusiveTagGroups.DefaultText, vocabulary);

        Assert.DoesNotContain(1, groups.ExcludedBy(0));
        Assert.DoesNotContain(2, groups.ExcludedBy(1));
        Assert.Contains(3, groups.ExcludedBy(0));
        Assert.Contains(4, groups.ExcludedBy(1));
        Assert.DoesNotContain(4, groups.ExcludedBy(3)); // 2girls implies multiple_girls
    }

    [Fact]
    public void Parse_SkipsCommentsAndUnknownTags()
    {
        var vocabulary = Vocabulary("a", "b");
        var groups = ExclusiveTagGroups.Parse("# a, b\na, nonexistent\n", vocabulary);

        Assert.True(groups.IsEmpty);
    }

    [Fact]
    public void Resolve_KeepsTheStrongestOfAContradictoryPair()
    {
        var vocabulary = Vocabulary("solo", "multiple girls", "hat");
        var groups = ExclusiveTagGroups.Parse("solo, multiple_girls", vocabulary);

        var kept = groups.Resolve([(Row: 1, Confidence: 0.9f), (Row: 2, Confidence: 0.8f), (Row: 0, Confidence: 0.7f)], c => c.Row);

        Assert.Equal([1, 2], kept.Select(c => c.Row));
    }
}
