using System.Collections.Concurrent;

namespace UnbooruTagger.Crawler;

/// <summary>
/// Shared site-failover bookkeeping for both <see cref="DatasetCrawler"/> and
/// <see cref="TagRefresher"/>: when a site's listing/download/lookup call exhausts
/// <see cref="TransientHttpRetry"/>'s retries and throws, drop that site for the rest
/// of the run and keep going with whatever's left, rather than taking the whole run
/// down over one site's outage — unless that was the last site, in which case there's
/// nothing left to make progress with. <see cref="ConcurrentDictionary{TKey,TValue}"/>
/// (not a plain <c>Dictionary</c>) because <see cref="DatasetCrawler"/> now runs one
/// worker task per site concurrently — each only ever writes its own key, but a plain
/// <c>Dictionary</c> isn't safe for a write on one thread happening alongside a read
/// (e.g. another site's "am I the last one left" check) on another, regardless of key.
/// </summary>
internal static class SiteAvailability
{
    /// <summary>
    /// Records <paramref name="site"/> as unavailable in <paramref name="unavailableSites"/>
    /// (site -&gt; failure reason) for the rest of this run, and — if every configured
    /// site in <paramref name="sites"/> is now unavailable — throws
    /// <see cref="AllSitesUnavailableException"/> instead of letting the caller grind
    /// through everything else with nothing left to fetch from.
    /// </summary>
    public static void MarkUnavailable(
        string site,
        Exception ex,
        IReadOnlyList<string> sites,
        ConcurrentDictionary<string, string> unavailableSites,
        Action<string>? report)
    {
        unavailableSites[site] = ex.Message;
        report?.Invoke($"'{site}' is unavailable ({ex.Message}) — continuing with the remaining site(s).");

        if (sites.All(unavailableSites.ContainsKey))
            throw new AllSitesUnavailableException(unavailableSites);
    }
}
