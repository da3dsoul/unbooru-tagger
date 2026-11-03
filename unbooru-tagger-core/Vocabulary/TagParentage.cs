using System.Text.Json;

namespace UnbooruTagger.Core.Vocabulary;

/// <summary>
/// The reviewed tag parentage rules (the tag-parentage review page's <c>rules.json</c>): which tags are a kind
/// of another (boots, geta and the *_footwear colours are shoes; hair_ribbon is a hair_ornament), which only
/// imply another's presence (footwear_bow → shoes: the bow's box isn't the shoe), and which rule another out
/// (no_shoes, barefoot → shoes). Only rules whose verdict is "accept" are kept; a rule nobody reviewed
/// counts by its draft. Names are raw booru names, matched to vocabulary identities the way
/// <see cref="TagImplications"/> matches them.
///
/// Where these differ from Danbooru's own conventions, the reviewed ones win: on Danbooru <c>shoes</c>
/// means non-boot shoes (only 9% of boot images carry it), and <c>hair_ornament</c> leaves out ribbons
/// and bows. <see cref="Bind"/> turns them into the per-row structure training and evaluation use.
/// </summary>
public sealed class TagParentage
{
    public const string FileName = "tag_parentage.json";

    public IReadOnlyList<(string Child, string Parent)> IsA { get; }
    public IReadOnlyList<(string Child, string Parent)> Implies { get; }
    public IReadOnlyList<(string Child, string Parent)> Excludes { get; }

    public int RuleCount => IsA.Count + Implies.Count + Excludes.Count;

    public static TagParentage Empty { get; } = new([], [], []);

    public TagParentage(IReadOnlyList<(string, string)> isA, IReadOnlyList<(string, string)> implies, IReadOnlyList<(string, string)> excludes)
    {
        IsA = isA;
        Implies = implies;
        Excludes = excludes;
    }

    /// <summary>Label completion: an image with the child also carries the parent, for both is-a and implies rules.</summary>
    public IEnumerable<(string Antecedent, string Consequent)> CompletionPairs => IsA.Concat(Implies);

    /// <summary>The rules file a <c>--tag-parentage</c> value names: the value itself, <c>none</c> for no rules, or (null) <paramref name="directory"/>'s saved copy.</summary>
    public static string? ResolvePath(string? option, string directory) => option switch
    {
        null => Path.Combine(directory, FileName),
        "none" => null,
        _ => option,
    };

    /// <summary>Loads a review-page <c>rules.json</c>; a missing file means no rules.</summary>
    public static TagParentage Load(string? path)
    {
        if (path is null || !File.Exists(path))
            return Empty;
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var isA = new List<(string, string)>();
        var implies = new List<(string, string)>();
        var excludes = new List<(string, string)>();
        foreach (var rule in document.RootElement.GetProperty("rules").EnumerateArray())
        {
            var verdict = rule.TryGetProperty("verdict", out var v) && v.ValueKind == JsonValueKind.String ? v.GetString()
                : rule.TryGetProperty("draft", out var d) ? d.GetString() : null;
            if (verdict != "accept")
                continue;
            var pair = (TagImplications.RawName(rule.GetProperty("child").GetString()!), TagImplications.RawName(rule.GetProperty("parent").GetString()!));
            switch (rule.GetProperty("kind").GetString())
            {
                case "is_a": isA.Add(pair); break;
                case "implies": implies.Add(pair); break;
                case "excludes": excludes.Add(pair); break;
            }
        }

        return new TagParentage(isA, implies, excludes);
    }

    /// <summary>
    /// The rules against <paramref name="vocabulary"/>, with Danbooru's own implications
    /// (<paramref name="danbooruPairs"/>, raw names) supplying the rest of each family: Danbooru already
    /// says platform_boots → boots, the review added boots → shoes.
    /// </summary>
    public TagFamilies Bind(TagVocabulary vocabulary, IEnumerable<(string Antecedent, string Consequent)> danbooruPairs)
    {
        var size = vocabulary.Records.Count;
        var rowByRaw = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var record in vocabulary.Records)
            rowByRaw.TryAdd(TagImplications.RawName(record.Tag), record.RowIndex);
        int? Row(string raw) => rowByRaw.TryGetValue(raw, out var row) ? row : null;

        // Kind-of closure: Danbooru's implications plus the reviewed is-a rules (not the implies rules: a
        // footwear_bow isn't a kind of shoe, so it doesn't name which shoe it is).
        var kindOf = TagImplications.FromRawPairs(danbooruPairs.Concat(IsA), vocabulary);
        var reviewedParents = new List<int>[size];
        foreach (var (child, parent) in IsA)
            if (Row(child) is { } c && Row(parent) is { } p && c != p)
                (reviewedParents[c] ??= []).Add(p);

        // A tag's family ancestors: the ancestors reached through a reviewed is-a rule from the tag or from one of
        // its ancestors. platform_boots → boots → shoes: shoes is one (boots → shoes was reviewed), boots isn't
        // (Danbooru's own rule, whose children keep Danbooru's meaning).
        var familyAncestors = new int[size][];
        var kinds = new List<int>[size];
        for (var t = 0; t < size; t++)
        {
            var found = new HashSet<int>();
            foreach (var start in kindOf.Implied(t).Prepend(t))
                if (reviewedParents[start] is { } parents)
                    foreach (var p in parents)
                    {
                        found.Add(p);
                        foreach (var above in kindOf.Implied(p))
                            found.Add(above);
                    }
            found.Remove(t);
            familyAncestors[t] = found.Order().ToArray();
            foreach (var a in kindOf.Implied(t))
                (kinds[a] ??= []).Add(t);
        }

        var childrenOf = new List<int>[size];
        for (var t = 0; t < size; t++)
            foreach (var a in familyAncestors[t])
                (childrenOf[a] ??= []).Add(t);

        var exclusions = new List<TagExclusion>[size];
        var excluders = new List<int>[size];
        foreach (var (child, parent) in Excludes)
        {
            if (Row(child) is not { } c || Row(parent) is not { } p || c == p)
                continue;
            var rows = (kinds[p] ?? []).Prepend(p).Where(r => r != c).Distinct().Order().ToArray();
            (exclusions[c] ??= []).Add(new TagExclusion(p, rows));
            foreach (var r in rows)
                (excluders[r] ??= []).Add(c);
        }

        static int[][] Sorted(List<int>[] lists) => lists.Select(l => l is null ? [] : l.Distinct().Order().ToArray()).ToArray();
        return new TagFamilies(
            Enumerable.Range(0, size).Select(t => kindOf.Implied(t).ToArray()).ToArray(),
            familyAncestors, Sorted(childrenOf), Sorted(kinds),
            exclusions.Select(l => l?.ToArray() ?? []).ToArray(), Sorted(excluders));
    }
}

/// <summary>An excludes rule on one row: the image certainly lacks <see cref="Rows"/> (the parent and every kind of it), unless it's tagged with <see cref="Parent"/>.</summary>
public sealed record TagExclusion(int Parent, int[] Rows);

/// <summary>
/// <see cref="TagParentage"/> bound to one vocabulary's rows. What a family means for training:
/// <list type="bullet">
/// <item><b>Unknown kind</b>: an image tagged with a family parent but with no kind of it named (shoes, but
/// not which shoes) says nothing about which kind it is. Every child of that parent is unknown there (weight 0),
/// never a negative: "it's a shoe" is right, and "boots" might be too. With a kind named (shoes and sandals),
/// the other kinds stay ordinary unverified negatives.</item>
/// <item><b>Exclusions</b>: an image tagged no_shoes certainly has no shoes, nor any kind of shoe, unless it is
/// also tagged shoes (often a second character). Certain negatives, and the best hard negatives there are: the
/// character usually wears them, the picture shows them without.</item>
/// </list>
/// </summary>
public sealed class TagFamilies
{
    private readonly int[][] _kindOf;
    private readonly int[][] _familyAncestors;
    private readonly int[][] _childrenOf;
    private readonly int[][] _kinds;
    private readonly TagExclusion[][] _exclusions;
    private readonly int[][] _excluders;

    internal TagFamilies(int[][] kindOf, int[][] familyAncestors, int[][] childrenOf, int[][] kinds, TagExclusion[][] exclusions, int[][] excluders)
    {
        _kindOf = kindOf;
        _familyAncestors = familyAncestors;
        _childrenOf = childrenOf;
        _kinds = kinds;
        _exclusions = exclusions;
        _excluders = excluders;
        IsEmpty = childrenOf.All(c => c.Length == 0) && exclusions.All(e => e.Length == 0);
    }

    public static TagFamilies Empty(int vocabularySize) =>
        new(Enumerable.Repeat(Array.Empty<int>(), vocabularySize).ToArray(),
            Enumerable.Repeat(Array.Empty<int>(), vocabularySize).ToArray(), Enumerable.Repeat(Array.Empty<int>(), vocabularySize).ToArray(),
            Enumerable.Repeat(Array.Empty<int>(), vocabularySize).ToArray(), Enumerable.Repeat(Array.Empty<TagExclusion>(), vocabularySize).ToArray(),
            Enumerable.Repeat(Array.Empty<int>(), vocabularySize).ToArray());

    public bool IsEmpty { get; }

    /// <summary>Every tag <paramref name="row"/> is a kind of, through Danbooru's or the reviewed is-a rules (platform_boots: boots, shoes).</summary>
    public IReadOnlyList<int> KindOf(int row) => Get(_kindOf, row);

    /// <summary>Family parents whose presence, with no kind named, leaves <paramref name="row"/> unknown.</summary>
    public IReadOnlyList<int> FamilyAncestors(int row) => Get(_familyAncestors, row);

    /// <summary>Rows <paramref name="row"/> is a family ancestor of.</summary>
    public IReadOnlyList<int> ChildrenOf(int row) => Get(_childrenOf, row);

    /// <summary>Every kind of <paramref name="row"/>: tags that imply it through Danbooru's or the reviewed is-a rules.</summary>
    public IReadOnlyList<int> Kinds(int row) => Get(_kinds, row);

    /// <summary>The excludes rules <paramref name="row"/> is the left side of.</summary>
    public IReadOnlyList<TagExclusion> Exclusions(int row) => (uint)row < (uint)_exclusions.Length ? _exclusions[row] : [];

    /// <summary>Tags whose images certainly lack <paramref name="row"/> (no_shoes for shoes and for boots).</summary>
    public IReadOnlyList<int> Excluders(int row) => Get(_excluders, row);

    /// <summary>Rows with at least one family ancestor: the only ones that can be unknown.</summary>
    public IEnumerable<int> FamilyRows => Enumerable.Range(0, _familyAncestors.Length).Where(t => _familyAncestors[t].Length > 0);

    /// <summary>
    /// True if <paramref name="tag"/>, absent from <paramref name="imageTags"/>, is an unnamed kind there: the image
    /// carries one of its family ancestors and no kind of that ancestor.
    /// </summary>
    public bool IsUnknown(IReadOnlyList<int> imageTags, int tag)
    {
        var ancestors = Get(_familyAncestors, tag);
        if (ancestors.Length == 0)
            return false;
        foreach (var a in ancestors)
            if (Contains(imageTags, a) && !NamesKind(imageTags, a))
                return true;
        return false;
    }

    /// <summary>Every row absent from <paramref name="imageTags"/> that <see cref="IsUnknown"/> there.</summary>
    public IEnumerable<int> UnknownRows(IReadOnlyList<int> imageTags)
    {
        foreach (var s in imageTags)
        {
            var children = Get(_childrenOf, s);
            if (children.Length == 0 || NamesKind(imageTags, s))
                continue;
            foreach (var t in children)
                if (!Contains(imageTags, t))
                    yield return t;
        }
    }

    /// <summary>Rows <paramref name="imageTags"/> certainly lack through an excludes rule (none tagged on the image itself).</summary>
    public IEnumerable<int> ExcludedRows(IReadOnlyList<int> imageTags)
    {
        foreach (var s in imageTags)
            foreach (var exclusion in Exclusions(s))
                if (!Contains(imageTags, exclusion.Parent))
                    foreach (var r in exclusion.Rows)
                        if (!Contains(imageTags, r))
                            yield return r;
    }

    private bool NamesKind(IReadOnlyList<int> imageTags, int parent)
    {
        var kinds = Get(_kinds, parent);
        if (kinds.Length == 0)
            return false;
        foreach (var r in imageTags)
            if (Array.BinarySearch(kinds, r) >= 0)
                return true;
        return false;
    }

    private static bool Contains(IReadOnlyList<int> rows, int row)
    {
        for (var i = 0; i < rows.Count; i++)
            if (rows[i] == row)
                return true;
        return false;
    }

    private static int[] Get(int[][] lists, int row) => (uint)row < (uint)lists.Length ? lists[row] : [];
}
