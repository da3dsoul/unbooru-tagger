using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Tests.Core;

public class TagImplicationsTests
{
    private static TagVocabulary Vocabulary(params string[] tags)
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        foreach (var tag in tags)
            vocabulary.AddTag(tag);
        return vocabulary;
    }

    [Fact]
    public void Complete_AddsImpliedTagsTransitively_MatchingPrefixedIdentitiesByRawName()
    {
        // 0 witch_hat -> 1 hat -> 2 headwear; 3 character:alt_costume -> 4 character:base
        var vocabulary = Vocabulary("witch_hat", "hat", "headwear", "character:alt_costume", "character:base", "other");
        var implications = TagImplications.FromRawPairs(
            [("witch_hat", "hat"), ("hat", "headwear"), ("alt_costume", "base"), ("not_in_vocab", "hat")], vocabulary);
        Assert.Equal(3, implications.RuleCount);

        var completed = implications.Complete([[0], [1, 2], [3, 5], [5]], out var added, out var changed);

        Assert.Equal([0, 1, 2], completed[0].Order());   // witch_hat -> hat -> headwear
        Assert.Equal([1, 2], completed[1].Order());      // already complete
        Assert.Equal([3, 4, 5], completed[2].Order());   // costume -> base character
        Assert.Equal([5], completed[3]);
        Assert.Equal(3, added);
        Assert.Equal(2, changed);
    }

    [Fact]
    public void FromRawPairs_SurvivesCycles()
    {
        var vocabulary = Vocabulary("a", "b");
        var implications = TagImplications.FromRawPairs([("a", "b"), ("b", "a")], vocabulary);
        var completed = implications.Complete([[0]], out _, out _);
        Assert.Equal([0, 1], completed[0].Order());
    }

    [Fact]
    public void Load_MissingFileMeansNoRules()
    {
        var implications = TagImplications.Load(Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.json"), Vocabulary("a"));
        Assert.Equal(0, implications.RuleCount);
    }
}
