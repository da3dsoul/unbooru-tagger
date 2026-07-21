namespace UnbooruTagger.Crawler;

/// <summary>
/// Thrown when every site configured for this run has failed — each one's listing or
/// download calls exhausted <see cref="TransientHttpRetry"/>'s retries and threw. A
/// single site failing is handled by dropping just that site and continuing with
/// whatever's left (see <see cref="DatasetCrawler"/>); this is the case where nothing
/// is left to continue with, so the run can't make any further progress.
/// </summary>
public sealed class AllSitesUnavailableException(IReadOnlyDictionary<string, string> failedSites)
    : Exception($"All configured sites failed: {string.Join(", ", failedSites.Select(kv => $"{kv.Key} ({kv.Value})"))}")
{
    public IReadOnlyDictionary<string, string> FailedSites { get; } = failedSites;
}
