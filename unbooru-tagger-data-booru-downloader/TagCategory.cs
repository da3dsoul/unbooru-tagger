namespace UnbooruTagger.Crawler;

/// <summary>
/// Danbooru's tag category scheme (<c>tags.json</c>'s <c>category</c> field), which
/// Gelbooru's dapi mirrors under the name <c>type</c> (same site heritage, same integer
/// codes — 2 is unused by either).
/// </summary>
public enum TagCategory
{
    General = 0,
    Artist = 1,
    Copyright = 3,
    Character = 4,
    Meta = 5,
}

/// <summary>
/// A tag's identity everywhere in this project — the vocabulary, <c>excluded_tags.txt</c>,
/// progress output, quota bookkeeping — is its raw booru name prefixed with its category,
/// computed once at survey time (see <see cref="DanbooruClient.ListTagsByCountDescendingAsync"/>/
/// <see cref="GelbooruClient.ListTagsByCountDescendingAsync"/>) and carried unchanged from
/// there on: <c>white_hair</c>, <c>character:frieren</c>, <c>series:sousou_no_frieren</c>.
/// <see cref="TagCategory.General"/> — the overwhelming majority of tags — is left
/// unprefixed since that's how the vast majority of tags read most naturally; every other
/// category gets a plain <c>category:name</c> string. Booru tag names never contain
/// <c>:</c> themselves (reserved for site search-query syntax), so this can never collide
/// with a real tag, and <see cref="RawName"/> can always recover the original by stripping
/// a recognized prefix.
/// </summary>
public static class TagCategoryNaming
{
    private static readonly (TagCategory Category, string Prefix)[] Prefixes =
    [
        (TagCategory.Artist, "artist:"),
        (TagCategory.Copyright, "series:"),
        (TagCategory.Character, "character:"),
        (TagCategory.Meta, "meta:"),
    ];

    public static string Identity(string rawName, TagCategory category)
    {
        foreach (var (candidate, prefix) in Prefixes)
            if (candidate == category)
                return prefix + rawName;

        return rawName;
    }

    /// <summary>Recovers the raw booru tag name a site's search API actually expects — the inverse of <see cref="Identity"/>.</summary>
    public static string RawName(string identity) => Split(identity).RawName;

    /// <summary>Recovers both halves of an identity string at once — the inverse of <see cref="Identity"/>.</summary>
    public static (string RawName, TagCategory Category) Split(string identity)
    {
        foreach (var (category, prefix) in Prefixes)
            if (identity.StartsWith(prefix, StringComparison.Ordinal))
                return (identity[prefix.Length..], category);

        return (identity, TagCategory.General);
    }

    /// <summary>Parses a site's raw category/type integer, defaulting unrecognized codes to <see cref="TagCategory.General"/> rather than guessing.</summary>
    public static TagCategory FromRawCode(int code) => Enum.IsDefined(typeof(TagCategory), code) ? (TagCategory)code : TagCategory.General;
}
