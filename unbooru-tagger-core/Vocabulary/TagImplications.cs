using System.Text.Json;

namespace UnbooruTagger.Core.Vocabulary;

/// <summary>
/// Danbooru's tag implications (witch_hat → hat, a costume tag → its character), loaded from
/// the crawler's <c>tag_implications.json</c> (<c>export-implications</c>) and used to fill in
/// the tags an image's own tags imply. Danbooru applies these rules to every post itself;
/// Gelbooru doesn't, so a Gelbooru-only image tagged witch_hat can lack hat. Unlike the
/// co-occurrence statistics, this is ground truth: the added tag is certainly correct.
/// The file holds raw booru names; vocabulary identities may carry a category prefix
/// (character:, series:, artist:, meta:), which is stripped to match.
/// </summary>
public sealed class TagImplications
{
    public const string FileName = "tag_implications.json";

    private static readonly string[] Prefixes = ["artist:", "series:", "character:", "meta:"];

    /// <summary>[row] → every row it implies, directly or through a chain (closure).</summary>
    private readonly int[][] _impliedBy;

    public int RuleCount { get; }

    private TagImplications(int[][] impliedBy, int ruleCount)
    {
        _impliedBy = impliedBy;
        RuleCount = ruleCount;
    }

    public static TagImplications Empty(int vocabularySize) => new(Enumerable.Repeat(Array.Empty<int>(), vocabularySize).ToArray(), 0);

    /// <summary>Loads <paramref name="path"/> against <paramref name="vocabulary"/>; rules whose tags aren't both in it are dropped. Missing file = no rules.</summary>
    public static TagImplications Load(string path, TagVocabulary vocabulary)
    {
        if (!File.Exists(path))
            return Empty(vocabulary.Records.Count);
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var pairs = document.RootElement.EnumerateArray()
            .Select(e => (e.GetProperty("Antecedent").GetString()!, e.GetProperty("Consequent").GetString()!));
        return FromRawPairs(pairs, vocabulary);
    }

    public static TagImplications FromRawPairs(IEnumerable<(string Antecedent, string Consequent)> pairs, TagVocabulary vocabulary)
    {
        var rowByRaw = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var record in vocabulary.Records)
            rowByRaw.TryAdd(Raw(record.Tag), record.RowIndex);

        var direct = new List<int>[vocabulary.Records.Count];
        var rules = 0;
        foreach (var (antecedent, consequent) in pairs)
        {
            if (!rowByRaw.TryGetValue(antecedent, out var from) || !rowByRaw.TryGetValue(consequent, out var to) || from == to)
                continue;
            (direct[from] ??= []).Add(to);
            rules++;
        }

        // Transitive closure per row (implication chains are short; a visited set guards cycles).
        var closure = new int[direct.Length][];
        for (var row = 0; row < direct.Length; row++)
        {
            if (direct[row] is null)
            {
                closure[row] = [];
                continue;
            }

            var seen = new HashSet<int>();
            var stack = new Stack<int>(direct[row]);
            while (stack.Count > 0)
            {
                var next = stack.Pop();
                if (next == row || !seen.Add(next))
                    continue;
                if (direct[next] is { } further)
                    foreach (var f in further)
                        stack.Push(f);
            }

            closure[row] = seen.Order().ToArray();
        }

        return new TagImplications(closure, rules);
    }

    /// <summary>Each image's tags plus every tag they imply; reports how many labels were added and to how many images.</summary>
    public List<List<int>> Complete(IReadOnlyList<IReadOnlyList<int>> imageTagRows, out long addedLabels, out int changedImages)
    {
        var result = new List<List<int>>(imageTagRows.Count);
        long added = 0;
        var changed = 0;
        foreach (var rows in imageTagRows)
        {
            List<int>? extra = null;
            foreach (var row in rows)
                if ((uint)row < (uint)_impliedBy.Length)
                    foreach (var implied in _impliedBy[row])
                        if (!rows.Contains(implied) && extra?.Contains(implied) != true)
                            (extra ??= []).Add(implied);

            var completed = new List<int>(rows.Count + (extra?.Count ?? 0));
            completed.AddRange(rows);
            if (extra is not null)
            {
                completed.AddRange(extra);
                added += extra.Count;
                changed++;
            }

            result.Add(completed);
        }

        addedLabels = added;
        changedImages = changed;
        return result;
    }

    private static string Raw(string tag)
    {
        foreach (var prefix in Prefixes)
            if (tag.StartsWith(prefix, StringComparison.Ordinal))
                return tag[prefix.Length..];
        return tag;
    }
}
