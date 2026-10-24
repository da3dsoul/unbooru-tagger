using System.Text;

namespace UnbooruTagger.Core.Encoding;

/// <summary>
/// A minimal BERT-style WordPiece tokenizer — the tokenizer most small frozen
/// sentence-embedding ONNX exports (the BERT/MiniLM family) expect, and the concrete
/// piece <see cref="OnnxTagTextEmbedder"/> deliberately left for a caller to supply
/// (see its own doc comment). Loads a standard "vocab.txt" (one token per line, line
/// number == token id — the same file any HuggingFace BERT-family tokenizer ships).
/// </summary>
public sealed class WordPieceTokenizer
{
    private const int MaxInputCharsPerWord = 200;
    private const string ContinuationPrefix = "##";

    private readonly IReadOnlyDictionary<string, long> _vocab;
    private readonly long _unknownTokenId;
    private readonly long? _startTokenId;
    private readonly long? _endTokenId;

    public WordPieceTokenizer(
        IReadOnlyDictionary<string, long> vocab,
        string unknownToken = "[UNK]",
        string startToken = "[CLS]",
        string endToken = "[SEP]")
    {
        _vocab = vocab;
        if (!_vocab.TryGetValue(unknownToken, out _unknownTokenId))
            throw new ArgumentException($"Vocabulary does not contain the unknown-token '{unknownToken}'.");

        // [CLS]/[SEP] are added around the sequence only if the vocabulary actually has
        // them -- a vocab built for a non-BERT-style model without those conventions
        // still works, it just tokenizes without the wrapper tokens.
        _startTokenId = _vocab.TryGetValue(startToken, out var start) ? start : null;
        _endTokenId = _vocab.TryGetValue(endToken, out var end) ? end : null;
    }

    /// <summary>Loads a "vocab.txt"-format file: one token per line, the line index (0-based) is that token's id.</summary>
    public static WordPieceTokenizer LoadVocabFile(string path, string unknownToken = "[UNK]", string startToken = "[CLS]", string endToken = "[SEP]")
    {
        var lines = File.ReadAllLines(path);
        var vocab = new Dictionary<string, long>(lines.Length);
        for (var i = 0; i < lines.Length; i++)
            vocab[lines[i]] = i;

        return new WordPieceTokenizer(vocab, unknownToken, startToken, endToken);
    }

    public long[] Tokenize(string text)
    {
        var ids = new List<long>();
        if (_startTokenId is { } start)
            ids.Add(start);

        foreach (var word in BasicTokenize(text))
            ids.AddRange(WordPieceTokenize(word));

        if (_endTokenId is { } end)
            ids.Add(end);

        return ids.ToArray();
    }

    /// <summary>Lowercases, then splits on whitespace and punctuation (each punctuation character becomes its own token) — the standard BERT "basic tokenizer" step that runs before WordPiece subword splitting.</summary>
    private static IEnumerable<string> BasicTokenize(string text)
    {
        var words = new List<string>();
        var current = new StringBuilder();

        void FlushCurrent()
        {
            if (current.Length > 0)
            {
                words.Add(current.ToString());
                current.Clear();
            }
        }

        foreach (var ch in text.ToLowerInvariant())
        {
            if (char.IsWhiteSpace(ch))
            {
                FlushCurrent();
            }
            else if (char.IsPunctuation(ch) || char.IsSymbol(ch))
            {
                FlushCurrent();
                words.Add(ch.ToString());
            }
            else
            {
                current.Append(ch);
            }
        }

        FlushCurrent();
        return words;
    }

    /// <summary>Greedy longest-match-first subword splitting: repeatedly takes the longest prefix of the remaining characters that's in the vocabulary (with a "##" prefix for every piece after the first), falling back to a single [UNK] for the whole word if no split covers it.</summary>
    private IEnumerable<long> WordPieceTokenize(string word)
    {
        if (word.Length > MaxInputCharsPerWord)
        {
            yield return _unknownTokenId;
            yield break;
        }

        var pieceIds = new List<long>();
        var start = 0;
        while (start < word.Length)
        {
            var end = word.Length;
            long? matchedId = null;
            while (end > start)
            {
                var candidate = start == 0 ? word[start..end] : ContinuationPrefix + word[start..end];
                if (_vocab.TryGetValue(candidate, out var id))
                {
                    matchedId = id;
                    break;
                }
                end--;
            }

            if (matchedId is null)
            {
                yield return _unknownTokenId;
                yield break;
            }

            pieceIds.Add(matchedId.Value);
            start = end;
        }

        foreach (var id in pieceIds)
            yield return id;
    }
}
