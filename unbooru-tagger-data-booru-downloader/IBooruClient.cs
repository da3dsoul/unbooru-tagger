namespace UnbooruTagger.Crawler;

/// <summary>
/// A thin client for one booru site's bulk tag/post listing endpoints. Deliberately not
/// <c>BooruSharp</c> (already a dependency of the sibling <c>unbooru</c> repo) — that
/// library is shaped around fetching a single post by id/md5, not bulk listing with
/// cursor pagination and a per-site rate limiter, which is what this project needs.
/// </summary>
public interface IBooruClient
{
    string SiteName { get; }

    /// <summary>The site's hard cap on posts per page (200 for Danbooru, 100 for Gelbooru regardless of any requested limit).</summary>
    int PageSize { get; }

    /// <summary>This client's own rate limiter, reused by <c>DatasetCrawler</c> for raw image downloads from the same site/CDN — those were previously unthrottled entirely, which is what let a burst of downloads trip the CDN's own 429s.</summary>
    IRateLimiter RateLimiter { get; }

    /// <summary>Streams every tag sorted by post count descending. Callers should stop enumerating once counts fall below their eligibility threshold — cheap because the ordering means nothing past that point is worth reading.</summary>
    IAsyncEnumerable<BooruTagCount> ListTagsByCountDescendingAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Lists one page of posts for <paramref name="tagQuery"/> (a plain tag name for a
    /// positive search, <c>-name</c> for the negative top-up's exclusion search).
    /// <paramref name="cursor"/> is <see langword="null"/> for the first page, otherwise
    /// the <see cref="BooruPostPage.NextCursor"/> this same client returned last call.
    /// </summary>
    Task<BooruPostPage> ListPostsAsync(string tagQuery, string? cursor, CancellationToken cancellationToken = default);
}
