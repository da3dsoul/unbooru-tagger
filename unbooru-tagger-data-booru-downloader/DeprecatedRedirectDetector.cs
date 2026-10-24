namespace UnbooruTagger.Crawler;

/// <summary>What <see cref="DeprecatedRedirectDetector"/> concluded about one (tag, site) probe — one row of the command's report.</summary>
public enum DeprecatedRedirectOutcome
{
    /// <summary>A candidate verified as an exact, ordered post-id match against the deprecated tag's own sample. The only outcome that ever writes an alias.</summary>
    Confirmed,

    /// <summary>
    /// The tag's raw name is already a known alias antecedent (an official Danbooru/Gelbooru
    /// alias, or one a previous run of this command already confirmed), so it's already
    /// handled correctly and re-verifying it live would be wasted network cost. Decided
    /// before any request is made — this outcome never costs a round-trip.
    /// </summary>
    AlreadyAliased,

    /// <summary>The site returned no posts at all for this tag — nothing to compare against, nothing to detect.</summary>
    NoResults,

    /// <summary>
    /// Fewer sampled posts than <see cref="DeprecatedRedirectOptions.MinSamplePosts"/>. An
    /// exact-sequence match over a handful of posts is not evidence of anything: two
    /// genuinely different tags that happen to sit on the same three images match trivially.
    /// </summary>
    SampleTooSmall,

    /// <summary>
    /// Every sampled post literally carries this tag, which is exactly how a NON-redirected
    /// tag search behaves (a genuine search for X returns only posts tagged X). Treated as
    /// proof the query was not redirected, and skipped before any verification request.
    /// See <see cref="DeprecatedRedirectDetector"/>'s own doc comment for why this is the
    /// single most valuable false-positive guard here.
    /// </summary>
    SearchLooksGenuine,

    /// <summary>No tag was present on 100% of the sampled posts, so there is no possible redirect target to verify.</summary>
    NoCandidates,

    /// <summary>
    /// More tags were present on 100% of the sampled posts than
    /// <see cref="DeprecatedRedirectOptions.MaxCandidates"/> allows verifying. Deliberately
    /// NOT truncated to an arbitrary prefix — a candidate set that large means the sample
    /// was too small/homogeneous to discriminate, and the fix is a bigger
    /// <c>--sample-pages</c>, not spending requests on an arbitrary subset.
    /// </summary>
    TooManyCandidates,

    /// <summary>Candidates existed but none of them reproduced the deprecated tag's own result sequence exactly. The expected, safe outcome for most tags — co-occurrence alone is not a redirect.</summary>
    NoCandidateVerified,

    /// <summary>More than one candidate reproduced the sequence exactly, so which one the site actually redirects to is genuinely undecidable from search results alone. Reported, never written.</summary>
    AmbiguousCandidates,

    /// <summary>A candidate verified, but writing it would close an alias cycle (X -&gt; ... -&gt; X) against the aliases already known. Reported, never written.</summary>
    WouldCreateCycle,

    /// <summary>
    /// A candidate verified, but so many OTHER antecedents in the same run verified against
    /// the very same target that the target cannot plausibly be the real redirect
    /// destination for all of them — see
    /// <see cref="DeprecatedRedirectOptions.MaxAntecedentsPerTarget"/>. Every confirmation
    /// pointing at such a target is downgraded to this, reported, and never written. A
    /// run-level conclusion, not a per-tag one: nothing about the individual probe looked
    /// wrong, which is exactly why it takes the whole run's shape to catch it.
    /// </summary>
    SuspectedHubTarget,
}

/// <summary>Tuning for <see cref="DeprecatedRedirectDetector.DetectAsync"/> — every value is a network-cost/confidence trade, see each member.</summary>
public sealed record DeprecatedRedirectOptions
{
    /// <summary>
    /// Pages of search results sampled per tag (and per candidate being verified). Two
    /// pages is ~200 posts on Danbooru (200/page) and ~200 on Gelbooru (100/page) — enough
    /// that an exact ordered post-id match is overwhelming evidence, and enough that the
    /// "present on 100% of sampled posts" candidate set collapses to a handful of tags
    /// instead of every common tag in the fandom. Verification is incremental (a candidate
    /// that already differs on page 1 is rejected without fetching page 2), so raising this
    /// costs far less than pages x candidates would suggest.
    /// </summary>
    public int SamplePages { get; init; } = 2;

    /// <summary>
    /// Minimum sampled posts before a match is trusted at all. An exact match over a very
    /// short sequence is cheap coincidence — two unrelated tags that happen to sit on the
    /// same handful of images produce it — so a tag with fewer results than this is left
    /// unresolved rather than guessed at.
    /// </summary>
    public int MinSamplePosts { get; init; } = 20;

    /// <summary>Cap on candidates verified for one (tag, site) probe, bounding worst-case requests per tag at <c>SamplePages * (1 + MaxCandidates)</c>. See <see cref="DeprecatedRedirectOutcome.TooManyCandidates"/>.</summary>
    public int MaxCandidates { get; init; } = 8;

    /// <summary>
    /// How many DISTINCT antecedents may confirm against one and the same target in a single
    /// run before every one of those confirmations is distrusted as a
    /// <see cref="DeprecatedRedirectOutcome.SuspectedHubTarget"/>. Strictly "more than", so a
    /// target sitting exactly on this number is still written — same off-by-one as
    /// <see cref="MaxCandidates"/>.
    ///
    /// This exists because per-tag verification, however exact, cannot see a site-wide
    /// FALLBACK. Confirmed live on Gelbooru: a query string it can't parse as a real search
    /// filter — <c>l</c>, <c>1:1</c>, <c>waifu</c>, ... — silently returns some shared default
    /// result set, byte-identical to a real search for <c>bad_tag</c>. Every such query
    /// therefore "verifies" against <c>bad_tag</c> with a genuinely exact, genuinely ordered
    /// post-id match; the match is real, the redirect is not. In the first full run this
    /// produced 1,000 of 3,635 confirmations (27.5%) all pointing at that one target.
    ///
    /// The default is calibrated against that run's actual distribution, which separates the
    /// two populations cleanly. The largest LEGITIMATE cluster was <c>ass</c> at 47
    /// antecedents (real typo/synonym variants — <c>nude_ass</c>, <c>bigassbooty</c>,
    /// <c>dat_ass</c>, <c>fat_butt</c>, spot-checked by hand), with <c>smile</c> next at 15
    /// and a smooth natural drop-off below that; the artifact sat alone at 1000, over 20x
    /// clear of it. 150 is ~3x above the largest observed real cluster (headroom for a
    /// hub-shaped but genuine tag this corpus hasn't hit yet) and ~7x below the observed
    /// artifact. It is a judgement call on ONE run's data, not a derived constant — hence
    /// <c>--max-antecedents-per-target</c>, so a future run whose real distribution reaches
    /// higher can be re-scoped without a code change.
    ///
    /// Deliberately biased low, in the same spirit as every other threshold here: suppressing
    /// a real hub costs a re-run with the option raised, while writing a fallback artifact
    /// permanently folds hundreds of unrelated tags into one identity across the vocabulary,
    /// the survey and every future crawl.
    /// </summary>
    public int MaxAntecedentsPerTarget { get; init; } = 150;
}

/// <summary>
/// One (tag, site) probe's conclusion. <paramref name="SiteName"/> is <see langword="null"/>
/// for <see cref="DeprecatedRedirectOutcome.AlreadyAliased"/>, which is decided without
/// contacting any site at all.
/// </summary>
public sealed record DeprecatedRedirectFinding(
    string TagName,
    string? SiteName,
    DeprecatedRedirectOutcome Outcome,
    string? Target,
    int SampledPosts,
    IReadOnlyList<string> Candidates,
    IReadOnlyList<string> VerifiedTargets,
    int RequestsMade);

/// <summary>Reports what <see cref="DeprecatedRedirectDetector"/> is doing right now, for a caller to render as a live progress bar — a full run is thousands of rate-limited round-trips and is otherwise entirely silent for hours.</summary>
public sealed class DeprecatedRedirectProgress
{
    /// <summary>(tags completed so far, tags total, the tag identity about to be probed).</summary>
    public Action<int, int, string>? OnTagStarted { get; init; }

    /// <summary>Fired once per (tag, site) probe, as soon as it concludes.</summary>
    public Action<DeprecatedRedirectFinding>? OnFinding { get; init; }
}

/// <summary>
/// Everything one <see cref="DeprecatedRedirectDetector.DetectAsync"/> run concluded.
/// <paramref name="NewAliases"/> holds ONLY the newly confirmed antecedent -&gt; consequent
/// pairs (raw names, same shape as <see cref="TagAliasCache"/>'s file) — it is not the
/// merged table, so a caller can report exactly what this run added and merge it itself.
/// </summary>
public sealed record DeprecatedRedirectDetectionResult(
    IReadOnlyList<DeprecatedRedirectFinding> Findings,
    IReadOnlyDictionary<string, string> NewAliases,
    int RequestsMade);

/// <summary>
/// Implements <c>detect-deprecated-redirects</c>: finds tags a site has deprecated and now
/// silently redirects a SEARCH for, with zero trace in any alias API.
///
/// This is a different failure from the one <c>prune-aliased-tags</c> handles. That one
/// covers tags Danbooru/Gelbooru have officially aliased away, which both sites publish
/// (<see cref="DanbooruClient.ListActiveTagAliasesAsync"/>/
/// <see cref="GelbooruClient.ListActiveTagAliasesAsync"/>) and which are therefore
/// discoverable by just reading their tables. The case here is invisible to both of those:
/// Gelbooru's <c>pussy_licking</c> (flagged deprecated via its own <c>type=6</c>, so
/// <see cref="TagSurveyResult.Deprecated"/> is set for it) returns byte-identical search
/// results to <c>cunnilingus</c> — same post count, same post-id sequence — while appearing
/// in no alias listing anywhere. Confirmed live. The only way to find one is empirically,
/// by comparing searches, which is what this does. The consequence of NOT finding one is
/// the same as for an unknown official alias: every post the site returns for the tag
/// genuinely carries the redirect TARGET's name, never the searched one, so the crawler
/// pages through the site's whole result set earning zero credit against that tag's quota.
///
/// The algorithm per deprecated+eligible tag X, on each site that has a surveyed count
/// for it:
/// <list type="number">
/// <item>Sample X's own search results (<see cref="DeprecatedRedirectOptions.SamplePages"/>
/// pages, id-descending — both clients already sort that way).</item>
/// <item>If EVERY sampled post literally carries X, stop: that is precisely how an
/// un-redirected tag search behaves, so the query provably wasn't redirected. This is the
/// strongest guard in the whole method, and it's free — it kills the one false-positive
/// mode that actually threatens this design (X being a genuine, narrower tag that happens
/// to only ever appear alongside some broader tag Y, whose results could otherwise coincide
/// with Y's own for a whole sample).</item>
/// <item>Candidates are the tags present on 100% of the sampled posts. That set is
/// guaranteed to contain the true target if there is one: a redirected search returns the
/// target's own posts, which all really carry the target's tag.</item>
/// <item>Verify each candidate Y by independently sampling Y's OWN search and requiring an
/// exact, ordered, page-by-page post-id match against X's sample — including matching
/// whether the result set was exhausted at the same point. Co-occurrence alone proves
/// nothing; only this turns "these tags travel together" into "these are the same
/// underlying query".</item>
/// <item>Exactly one verified candidate is a confirmed redirect. Zero (the common case) or
/// more than one leaves X unresolved, which is a safe, expected outcome rather than an
/// error.</item>
/// </list>
///
/// Then, once every tag has been probed, one RUN-level pass over the findings —
/// <see cref="DeprecatedRedirectOutcome.SuspectedHubTarget"/> — throws out any target that
/// too many distinct antecedents converged on. That is the one false positive per-tag
/// verification structurally cannot catch, because the exact match it demands is really
/// there: a site that answers an unparseable query with a shared fallback result set makes
/// every such query verify against whatever real tag happens to return that same set. It
/// costs no extra request, since the convergence itself is the evidence.
///
/// Deliberate conservatism throughout, because a wrong entry in <c>tag_aliases.json</c> is
/// far more expensive than a missed one: a false alias permanently folds two genuinely
/// different tags into one identity across the vocabulary, the survey and every future
/// crawl, whereas a miss just means re-running this later. Every ambiguous, undersized or
/// unverifiable case therefore reports and writes nothing.
///
/// Known residual uncertainty, in the same spirit: a site adding a new post to Y in between
/// sampling X and sampling Y shifts Y's id-descending page and breaks the exact match. That
/// is a false NEGATIVE (a missed detection), never a false positive, and re-running picks
/// it up — so it is left unhandled rather than papered over with an id anchor, which would
/// weaken the comparison in exchange.
/// </summary>
public static class DeprecatedRedirectDetector
{
    /// <param name="tags">
    /// The tags to probe — the caller's job to have already filtered these to currently
    /// deprecated AND currently eligible rows (eligibility is recomputed live from
    /// <c>--min-images</c> + <c>excluded_tags.txt</c> everywhere in this codebase, never
    /// read back out of <c>crawl.sqlite</c>'s stored <c>Eligible</c> column, and this is no
    /// different).
    /// </param>
    /// <param name="knownAliases">
    /// Antecedent raw name -&gt; consequent raw name, exactly as <see cref="TagAliasCache"/>
    /// stores it. Any tag whose raw name is already a key here is skipped outright with
    /// <see cref="DeprecatedRedirectOutcome.AlreadyAliased"/> and costs no request: it is
    /// already correctly handled by the official alias machinery, and a second, empirically
    /// derived entry for the same antecedent could only duplicate or contradict it.
    /// </param>
    public static async Task<DeprecatedRedirectDetectionResult> DetectAsync(
        IReadOnlyList<TagSurveyResult> tags,
        IReadOnlyDictionary<string, IBooruClient> clients,
        IReadOnlyDictionary<string, string>? knownAliases = null,
        DeprecatedRedirectOptions? options = null,
        DeprecatedRedirectProgress? progress = null,
        CancellationToken cancellationToken = default)
    {
        var opts = options ?? new DeprecatedRedirectOptions();

        // Seeded with the aliases already known and grown as this run confirms more, so
        // both the "already an antecedent" skip and the cycle guard below see every entry
        // that would exist on disk after this run — not just the ones that existed before it.
        var workingAliases = new Dictionary<string, string>(StringComparer.Ordinal);
        if (knownAliases is not null)
            foreach (var (antecedent, consequent) in knownAliases)
                workingAliases[antecedent] = consequent;

        var newAliases = new Dictionary<string, string>(StringComparer.Ordinal);
        var findings = new List<DeprecatedRedirectFinding>();
        var requestsMade = 0;

        // Danbooru first, matching TagAliasCache.FetchAndCacheAsync's own "Danbooru wins on
        // overlap" tie-break: the alias table is one flat map keyed by antecedent, so if
        // both sites redirect the same tag (to possibly different targets) only one answer
        // can be stored, and it should be the same site's answer everywhere else in this
        // codebase prefers. Once one site confirms, the remaining sites are skipped for that
        // tag entirely — their answer could not be stored anyway, so probing them would be
        // pure wasted network.
        var orderedSites = clients.Keys
            .OrderBy(site => string.Equals(site, "danbooru", StringComparison.Ordinal) ? 0 : 1)
            .ThenBy(site => site, StringComparer.Ordinal)
            .ToList();

        for (var index = 0; index < tags.Count; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var tag = tags[index];
            progress?.OnTagStarted?.Invoke(index, tags.Count, tag.Name);

            var rawName = TagCategoryNaming.RawName(tag.Name);

            if (workingAliases.ContainsKey(rawName))
            {
                var skipped = new DeprecatedRedirectFinding(tag.Name, null, DeprecatedRedirectOutcome.AlreadyAliased, workingAliases[rawName], 0, [], [], 0);
                findings.Add(skipped);
                progress?.OnFinding?.Invoke(skipped);
                continue;
            }

            foreach (var site in orderedSites)
            {
                if (!SurveyedOnSite(tag, site))
                    continue;

                var finding = await ProbeAsync(tag, rawName, clients[site], workingAliases, opts, cancellationToken).ConfigureAwait(false);
                requestsMade += finding.RequestsMade;
                findings.Add(finding);
                progress?.OnFinding?.Invoke(finding);

                if (finding.Outcome != DeprecatedRedirectOutcome.Confirmed)
                    continue;

                workingAliases[rawName] = finding.Target!;
                newAliases[rawName] = finding.Target!;
                break;
            }
        }

        SuppressHubTargets(findings, newAliases, opts.MaxAntecedentsPerTarget);

        return new DeprecatedRedirectDetectionResult(findings, newAliases, requestsMade);
    }

    /// <summary>
    /// The run-level guard per-tag verification structurally cannot be: rewrites every
    /// <see cref="DeprecatedRedirectOutcome.Confirmed"/> finding whose target absorbed more
    /// than <paramref name="maxAntecedentsPerTarget"/> DISTINCT antecedents this run into
    /// <see cref="DeprecatedRedirectOutcome.SuspectedHubTarget"/>, and drops it from
    /// <paramref name="newAliases"/> so nothing is written for it. See
    /// <see cref="DeprecatedRedirectOptions.MaxAntecedentsPerTarget"/> for the failure mode
    /// and the number.
    ///
    /// Purely a re-reading of findings already collected — it issues no request and cannot,
    /// by construction (it is handed no client). That is the entire point: "1,000 unrelated
    /// tags converged on one name" is itself conclusive evidence of a fallback, and asking
    /// the network again would only re-confirm the same exact match that caused the problem.
    ///
    /// Counted over distinct RAW names rather than tag identities, because raw names are what
    /// <c>tag_aliases.json</c> is keyed by — two identities sharing a raw name would collapse
    /// to a single alias entry, so they are a single antecedent as far as the damage goes.
    ///
    /// Grouped by target alone rather than by (site, target) for the same reason: the alias
    /// table has no site dimension, so a target's blast radius is the union across sites.
    ///
    /// One knowingly-accepted impurity: the run's cycle guard already saw these confirmations
    /// while they were still live, so a LATER tag could have been reported
    /// <see cref="DeprecatedRedirectOutcome.WouldCreateCycle"/> against a mapping this pass
    /// then withdrew. That direction only ever withholds a write, never adds a wrong one, and
    /// a re-run resolves it — unlike suppressing hubs mid-run, which would need the whole
    /// run's shape before the run is over.
    ///
    /// <see cref="DeprecatedRedirectProgress.OnFinding"/> is deliberately NOT re-fired for a
    /// downgraded finding: it is documented as firing once per probe, and a probe's
    /// conclusion changing after the fact is a property of the RUN, which the returned
    /// <see cref="DeprecatedRedirectDetectionResult"/> already carries in full.
    /// </summary>
    private static void SuppressHubTargets(
        List<DeprecatedRedirectFinding> findings,
        Dictionary<string, string> newAliases,
        int maxAntecedentsPerTarget)
    {
        var antecedentsByTarget = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
        foreach (var finding in findings)
        {
            if (finding.Outcome != DeprecatedRedirectOutcome.Confirmed)
                continue;

            if (!antecedentsByTarget.TryGetValue(finding.Target!, out var antecedents))
                antecedentsByTarget[finding.Target!] = antecedents = new HashSet<string>(StringComparer.Ordinal);
            antecedents.Add(TagCategoryNaming.RawName(finding.TagName));
        }

        var hubs = antecedentsByTarget
            .Where(pair => pair.Value.Count > maxAntecedentsPerTarget)
            .Select(pair => pair.Key)
            .ToHashSet(StringComparer.Ordinal);
        if (hubs.Count == 0)
            return;

        for (var index = 0; index < findings.Count; index++)
        {
            var finding = findings[index];
            if (finding.Outcome != DeprecatedRedirectOutcome.Confirmed || !hubs.Contains(finding.Target!))
                continue;

            findings[index] = finding with { Outcome = DeprecatedRedirectOutcome.SuspectedHubTarget };
            newAliases.Remove(TagCategoryNaming.RawName(finding.TagName));
        }
    }

    /// <summary>
    /// Whether the last survey actually recorded a count for this tag on this site — the
    /// only per-site signal <c>crawl.sqlite</c> keeps, since <see cref="TagSurveyResult.Deprecated"/>
    /// is already OR-merged across sites by <see cref="TagSurveyor"/> and can't say WHICH
    /// site flagged it. Probing only sites with a count is a strict superset of "the sites
    /// that could have flagged it deprecated" — the merged flag can only have come from a
    /// site that contributed a count in the first place — so nothing is missed by it, while
    /// a tag only one site knows about costs one probe instead of two. An unrecognized site
    /// name (nothing but danbooru/gelbooru exists today) is probed rather than silently
    /// skipped.
    /// </summary>
    private static bool SurveyedOnSite(TagSurveyResult tag, string site) => site switch
    {
        "danbooru" => tag.DanbooruCount is not null,
        "gelbooru" => tag.GelbooruCount is not null,
        _ => true,
    };

    private static async Task<DeprecatedRedirectFinding> ProbeAsync(
        TagSurveyResult tag,
        string rawName,
        IBooruClient client,
        IReadOnlyDictionary<string, string> workingAliases,
        DeprecatedRedirectOptions options,
        CancellationToken cancellationToken)
    {
        var requests = 0;

        DeprecatedRedirectFinding Finding(DeprecatedRedirectOutcome outcome, string? target, int sampledPosts, IReadOnlyList<string> candidates, IReadOnlyList<string> verified) =>
            new(tag.Name, client.SiteName, outcome, target, sampledPosts, candidates, verified, requests);

        var (pages, exhausted, posts, sampleRequests) = await SampleAsync(client, rawName, options.SamplePages, cancellationToken).ConfigureAwait(false);
        requests += sampleRequests;

        if (posts.Count == 0)
            return Finding(DeprecatedRedirectOutcome.NoResults, null, 0, [], []);
        if (posts.Count < options.MinSamplePosts)
            return Finding(DeprecatedRedirectOutcome.SampleTooSmall, null, posts.Count, [], []);

        // Tags present on 100% of the sample. Computed INCLUDING the searched tag itself,
        // because whether it survives the intersection is the whole point of the next check.
        var shared = new HashSet<string>(posts[0].Tags, StringComparer.Ordinal);
        foreach (var post in posts.Skip(1))
            shared.IntersectWith(post.Tags);

        if (shared.Contains(rawName))
            return Finding(DeprecatedRedirectOutcome.SearchLooksGenuine, null, posts.Count, [], []);

        var candidates = shared.OrderBy(name => name, StringComparer.Ordinal).ToList();
        if (candidates.Count == 0)
            return Finding(DeprecatedRedirectOutcome.NoCandidates, null, posts.Count, candidates, []);
        if (candidates.Count > options.MaxCandidates)
            return Finding(DeprecatedRedirectOutcome.TooManyCandidates, null, posts.Count, candidates, []);

        var verified = new List<string>();
        foreach (var candidate in candidates)
        {
            var (matches, verifyRequests) = await VerifyAsync(client, candidate, pages, exhausted, cancellationToken).ConfigureAwait(false);
            requests += verifyRequests;
            if (matches)
                verified.Add(candidate);
        }

        if (verified.Count == 0)
            return Finding(DeprecatedRedirectOutcome.NoCandidateVerified, null, posts.Count, candidates, verified);
        if (verified.Count > 1)
            return Finding(DeprecatedRedirectOutcome.AmbiguousCandidates, null, posts.Count, candidates, verified);

        var target = verified[0];
        return WouldCreateCycle(workingAliases, rawName, target)
            ? Finding(DeprecatedRedirectOutcome.WouldCreateCycle, target, posts.Count, candidates, verified)
            : Finding(DeprecatedRedirectOutcome.Confirmed, target, posts.Count, candidates, verified);
    }

    /// <summary>
    /// Pages one tag's search results, returning each page's post-id sequence separately
    /// (page boundaries are part of what gets compared — see <see cref="VerifyAsync"/>),
    /// the flattened posts, and whether the result set ran out before the page budget did.
    /// A page can legitimately come back with zero usable posts and still have a next cursor
    /// (both clients filter out posts missing md5/file_url — e.g. login-gated content), so
    /// pagination is driven by the cursor, never by a page looking empty.
    /// </summary>
    private static async Task<(List<IReadOnlyList<long>> Pages, bool Exhausted, List<BooruPost> Posts, int Requests)> SampleAsync(
        IBooruClient client, string tagQuery, int maxPages, CancellationToken cancellationToken)
    {
        var pages = new List<IReadOnlyList<long>>();
        var posts = new List<BooruPost>();
        var requests = 0;
        var exhausted = false;
        string? cursor = null;

        for (var page = 0; page < maxPages; page++)
        {
            var result = await client.ListPostsAsync(tagQuery, cursor, cancellationToken).ConfigureAwait(false);
            requests++;
            pages.Add(result.Posts.Select(post => post.PostId).ToList());
            posts.AddRange(result.Posts);

            cursor = result.NextCursor;
            if (cursor is null)
            {
                exhausted = true;
                break;
            }
        }

        return (pages, exhausted, posts, requests);
    }

    /// <summary>
    /// Re-runs the sample against <paramref name="candidate"/> and requires it to reproduce
    /// <paramref name="pages"/> exactly: same number of pages, same post ids in the same
    /// order within each page, and the same point of exhaustion. Anything less is not
    /// evidence of a redirect — two tags can share every post in a window and still be
    /// different queries the moment the window widens.
    ///
    /// Fetched incrementally and abandoned the instant a page differs, so the overwhelming
    /// majority of candidates (which are ordinary co-occurring tags) cost exactly one
    /// request to reject rather than a full sample.
    ///
    /// Matching the exhaustion point matters as much as matching the ids: if the searched
    /// tag ran out of results at the end of page 2 and the candidate has a third page, they
    /// are provably different result sets however identical the first two pages looked.
    /// </summary>
    private static async Task<(bool Matches, int Requests)> VerifyAsync(
        IBooruClient client, string candidate, IReadOnlyList<IReadOnlyList<long>> pages, bool exhausted, CancellationToken cancellationToken)
    {
        var requests = 0;
        string? cursor = null;

        for (var page = 0; page < pages.Count; page++)
        {
            var result = await client.ListPostsAsync(candidate, cursor, cancellationToken).ConfigureAwait(false);
            requests++;

            if (!result.Posts.Select(post => post.PostId).SequenceEqual(pages[page]))
                return (false, requests);

            cursor = result.NextCursor;
            var isLastPage = page == pages.Count - 1;

            // The candidate ran out of results while the searched tag still had more pages.
            if (!isLastPage && cursor is null)
                return (false, requests);

            // ...or the searched tag ran out here and the candidate did not.
            if (isLastPage && exhausted && cursor is not null)
                return (false, requests);
        }

        return (true, requests);
    }

    /// <summary>
    /// Whether adding <paramref name="antecedent"/> -&gt; <paramref name="consequent"/> to
    /// <paramref name="aliases"/> would close a cycle. <see cref="TagRowMutations.ResolveAlias"/>
    /// already walks alias chains cycle-safely at READ time, so a cycle wouldn't hang
    /// anything — but it would silently resolve to whichever arbitrary tag the walk happens
    /// to stop on, which is exactly the class of quiet, hard-to-trace tag-identity bug this
    /// codebase has been bitten by before. Cheaper to never write one.
    ///
    /// A pre-existing cycle that doesn't involve <paramref name="antecedent"/> is not this
    /// entry's fault and doesn't block it — the walk just has to terminate on it rather than
    /// spin, hence the visited set.
    /// </summary>
    internal static bool WouldCreateCycle(IReadOnlyDictionary<string, string> aliases, string antecedent, string consequent)
    {
        var visited = new HashSet<string>(StringComparer.Ordinal);
        var current = consequent;
        while (true)
        {
            if (string.Equals(current, antecedent, StringComparison.Ordinal))
                return true;
            if (!visited.Add(current))
                return false;
            if (!aliases.TryGetValue(current, out var next))
                return false;
            current = next;
        }
    }
}
