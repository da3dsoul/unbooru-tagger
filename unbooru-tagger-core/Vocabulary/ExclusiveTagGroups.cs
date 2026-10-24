namespace UnbooruTagger.Core.Vocabulary;

/// <summary>
/// Sets of tags that can never legitimately appear on the same image (solo vs.
/// multiple_girls, 1girl vs. 2girls, ...). Used two ways: training treats "the image has
/// one member, so every other member is absent" as a certain negative instead of an
/// unverified one (an untagged booru tag is usually just a missing tag, not proof of
/// absence, but an exclusive partner of a tag that IS present really is absent), and
/// inference keeps only the strongest member of a group instead of reporting
/// contradictory tags together.
///
/// File format: one group per line, members separated by commas, every pair within a
/// line mutually exclusive. Blank lines and lines starting with <c>#</c> are ignored.
/// Underscores and spaces are interchangeable (booru-style <c>multiple_girls</c> matches
/// the vocabulary's <c>multiple girls</c>). Tags not in the vocabulary are skipped.
/// </summary>
public sealed class ExclusiveTagGroups
{
    public const string FileName = "exclusive_tags.txt";

    /// <summary>The people-count and rating groups every booru agrees on — used when no file is given.</summary>
    public const string DefaultText =
        """
        # One group per line; every pair of tags on a line is mutually exclusive.
        1girl, 2girls, 3girls, 4girls, 5girls, 6+girls
        1girl, multiple_girls
        solo, multiple_girls
        solo, 2girls, 3girls, 4girls, 5girls, 6+girls
        1boy, 2boys, 3boys, 4boys, 5boys, 6+boys
        1boy, multiple_boys
        solo, multiple_boys
        solo, 2boys, 3boys, 4boys, 5boys, 6+boys
        no_humans, solo
        no_humans, 1girl
        no_humans, 1boy
        no_humans, multiple_girls
        no_humans, multiple_boys
        rating:general, rating:sensitive, rating:questionable, rating:explicit
        rating:safe, rating:questionable, rating:explicit
        """;

    private static readonly int[] None = [];

    private readonly Dictionary<int, int[]> _excludedByRow;

    public string SourceText { get; }

    private ExclusiveTagGroups(Dictionary<int, int[]> excludedByRow, string sourceText)
    {
        _excludedByRow = excludedByRow;
        SourceText = sourceText;
    }

    public static ExclusiveTagGroups Empty { get; } = new([], "");

    /// <summary>Every tag row that can't co-occur with <paramref name="row"/>.</summary>
    public IReadOnlyList<int> ExcludedBy(int row) => _excludedByRow.GetValueOrDefault(row, None);

    public bool IsEmpty => _excludedByRow.Count == 0;

    /// <summary>Every tag row that belongs to at least one group.</summary>
    public IEnumerable<int> Rows => _excludedByRow.Keys;

    public static ExclusiveTagGroups Parse(string text, TagVocabulary vocabulary)
    {
        var excluded = new Dictionary<int, HashSet<int>>();
        foreach (var rawLine in text.Split('\n'))
        {
            var line = rawLine.Trim();
            if (line.Length == 0 || line.StartsWith('#'))
                continue;

            var rows = line.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
                .Select(tag => TryResolve(vocabulary, tag))
                .OfType<int>()
                .Distinct()
                .ToList();

            foreach (var a in rows)
            foreach (var b in rows)
            {
                if (a == b)
                    continue;
                if (!excluded.TryGetValue(a, out var set))
                    excluded[a] = set = [];
                set.Add(b);
            }
        }

        return new ExclusiveTagGroups(
            excluded.ToDictionary(kv => kv.Key, kv => kv.Value.Order().ToArray()),
            text);
    }

    /// <summary>Loads <paramref name="path"/> if it exists, otherwise <see cref="DefaultText"/> when <paramref name="fallBackToDefaults"/> is set, otherwise <see cref="Empty"/>.</summary>
    public static ExclusiveTagGroups Load(string? path, TagVocabulary vocabulary, bool fallBackToDefaults)
    {
        if (path is not null && File.Exists(path))
            return Parse(File.ReadAllText(path), vocabulary);
        return fallBackToDefaults ? Parse(DefaultText, vocabulary) : Empty;
    }

    public void Save(string path) => File.WriteAllText(path, SourceText);

    /// <summary>
    /// Greedy resolution over candidates already sorted by descending confidence: keeps a
    /// candidate unless a stronger, already-kept candidate excludes it. Returns the kept
    /// candidates in their original order.
    /// </summary>
    public List<T> Resolve<T>(IReadOnlyList<T> candidatesByDescendingConfidence, Func<T, int> rowOf)
    {
        var kept = new List<T>();
        var blocked = new HashSet<int>();
        foreach (var candidate in candidatesByDescendingConfidence)
        {
            var row = rowOf(candidate);
            if (blocked.Contains(row))
                continue;

            kept.Add(candidate);
            foreach (var excludedRow in ExcludedBy(row))
                blocked.Add(excludedRow);
        }

        return kept;
    }

    private static int? TryResolve(TagVocabulary vocabulary, string tag)
    {
        if (vocabulary.TryGet(tag, out var record))
            return record.RowIndex;
        if (vocabulary.TryGet(tag.Replace('_', ' '), out record))
            return record.RowIndex;
        if (vocabulary.TryGet(tag.Replace(' ', '_'), out record))
            return record.RowIndex;
        return null;
    }
}
