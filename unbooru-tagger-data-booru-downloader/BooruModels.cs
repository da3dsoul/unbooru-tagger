namespace UnbooruTagger.Crawler;

/// <summary>One post as returned by either site's post-listing endpoint — only the fields this project actually uses.</summary>
public sealed record BooruPost(
    long PostId,
    string Md5,
    Uri FileUrl,
    IReadOnlyList<string> Tags,
    string Rating,
    DateTimeOffset CreatedAt,
    int Width,
    int Height);

/// <summary>
/// A tag name and its post count, as returned by a site's tag-listing endpoint.
/// <paramref name="Deprecated"/> is Danbooru's own <c>is_deprecated</c> flag, or (on
/// Gelbooru, which has no separate boolean) whether the site's own <c>type</c> code was
/// its dedicated deprecated-tag category (6) — see <see cref="GelbooruClient.ListTagsByCountDescendingAsync"/>.
/// </summary>
public sealed record BooruTagCount(string Name, int PostCount, bool Deprecated = false);

/// <summary>
/// A resolved tag alias: searching/tagging <paramref name="Antecedent"/> on this site is
/// silently redirected to <paramref name="Consequent"/>. Both are raw (un-prefixed,
/// site-local) tag names, never a <see cref="TagCategoryNaming"/> identity — see
/// <see cref="TagSurveyor"/> for why that matters.
/// </summary>
public sealed record BooruTagAlias(string Antecedent, string Consequent);

/// <summary>
/// A tag implication: every post tagged <paramref name="Antecedent"/> is also <paramref name="Consequent"/>
/// (witch_hat → hat). Raw site-local names, like <see cref="BooruTagAlias"/>.
/// </summary>
public sealed record BooruTagImplication(string Antecedent, string Consequent);

/// <summary>
/// One page of posts plus an opaque continuation token for the next page (null once
/// exhausted). The token's shape is site-specific (a "before this post id" cursor for
/// Danbooru, a page-index for Gelbooru) — callers should treat it as opaque and just
/// thread it back into the next call for the same site/tag.
/// </summary>
public sealed record BooruPostPage(IReadOnlyList<BooruPost> Posts, string? NextCursor);
