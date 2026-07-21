namespace UnbooruTagger.Crawler;

/// <summary>
/// Shared site-failover bookkeeping for both <see cref="DatasetCrawler"/> and
/// <see cref="TagRefresher"/>: when a site's listing/download/lookup call exhausts
/// <see cref="TransientHttpRetry"/>'s retries and throws, drop that site for the rest
/// of the run and keep going with whatever's left, rather than taking the whole run
/// down over one site's outage — unless that was the last site, in which case there's
/// nothing left to make progress with.
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
        Dictionary<string, string> unavailableSites,
        Action<string>? report)
    {
        unavailableSites[site] = ex.Message;
        report?.Invoke($"'{site}' is unavailable ({ex.Message}) — continuing with the remaining site(s).");

        if (sites.All(unavailableSites.ContainsKey))
            throw new AllSitesUnavailableException(unavailableSites);
    }
}
