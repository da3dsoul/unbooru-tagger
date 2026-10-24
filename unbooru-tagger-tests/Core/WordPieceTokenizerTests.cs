using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Tests.Core;

public class WordPieceTokenizerTests
{
    private static readonly Dictionary<string, long> BaseVocab = new()
    {
        ["[UNK]"] = 0,
        ["[CLS]"] = 1,
        ["[SEP]"] = 2,
        ["hatsune"] = 3,
        ["miku"] = 4,
        ["nurse"] = 5,
        ["costume"] = 6,
        ["cos"] = 7,
        ["##tume"] = 8,
        ["("] = 9,
        [")"] = 10,
    };

    [Fact]
    public void Tokenize_WrapsWithStartAndEndTokensWhenPresentInVocab()
    {
        var tokenizer = new WordPieceTokenizer(BaseVocab);

        var ids = tokenizer.Tokenize("miku");

        Assert.Equal(new long[] { 1, 4, 2 }, ids);
    }

    [Fact]
    public void Tokenize_MatchesWholeWordsGreedily()
    {
        var tokenizer = new WordPieceTokenizer(BaseVocab);

        var ids = tokenizer.Tokenize("hatsune miku nurse costume");

        Assert.Equal(new long[] { 1, 3, 4, 5, 6, 2 }, ids);
    }

    [Fact]
    public void Tokenize_SplitsAnUnknownWordIntoSubwordPiecesWithContinuationPrefix()
    {
        // "costume" itself is in the vocab (see the test above), but if it weren't, the
        // greedy longest-match should still fall back to "cos" + "##tume".
        var vocabWithoutWholeWord = new Dictionary<string, long>(BaseVocab);
        vocabWithoutWholeWord.Remove("costume");
        var tokenizer = new WordPieceTokenizer(vocabWithoutWholeWord);

        var ids = tokenizer.Tokenize("costume");

        Assert.Equal(new long[] { 1, 7, 8, 2 }, ids);
    }

    [Fact]
    public void Tokenize_FallsBackToASingleUnknownTokenWhenNoSplitCoversTheWord()
    {
        var tokenizer = new WordPieceTokenizer(BaseVocab);

        var ids = tokenizer.Tokenize("xenoverse");

        Assert.Equal(new long[] { 1, 0, 2 }, ids);
    }

    [Fact]
    public void Tokenize_SplitsPunctuationIntoItsOwnTokens()
    {
        var tokenizer = new WordPieceTokenizer(BaseVocab);

        var ids = tokenizer.Tokenize("miku(hatsune)");

        Assert.Equal(new long[] { 1, 4, 9, 3, 10, 2 }, ids);
    }

    [Fact]
    public void Tokenize_IsCaseInsensitive()
    {
        var tokenizer = new WordPieceTokenizer(BaseVocab);

        var ids = tokenizer.Tokenize("MIKU");

        Assert.Equal(new long[] { 1, 4, 2 }, ids);
    }

    [Fact]
    public void Tokenize_OmitsStartAndEndTokensWhenTheVocabDoesNotHaveThem()
    {
        var vocabWithoutSpecialTokens = new Dictionary<string, long> { ["[UNK]"] = 0, ["miku"] = 1 };
        var tokenizer = new WordPieceTokenizer(vocabWithoutSpecialTokens);

        var ids = tokenizer.Tokenize("miku");

        Assert.Equal(new long[] { 1 }, ids);
    }

    [Fact]
    public void Constructor_ThrowsWhenTheVocabHasNoUnknownToken()
    {
        var vocabWithoutUnknown = new Dictionary<string, long> { ["miku"] = 0 };

        Assert.Throws<ArgumentException>(() => new WordPieceTokenizer(vocabWithoutUnknown));
    }

    [Fact]
    public void LoadVocabFile_AssignsIdsByLineIndex()
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.txt");
        try
        {
            File.WriteAllLines(path, ["[UNK]", "[CLS]", "[SEP]", "miku"]);

            var tokenizer = WordPieceTokenizer.LoadVocabFile(path);
            var ids = tokenizer.Tokenize("miku");

            Assert.Equal(new long[] { 1, 3, 2 }, ids);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
