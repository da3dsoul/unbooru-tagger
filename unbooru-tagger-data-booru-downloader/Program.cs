using System.CommandLine;
using Spectre.Console;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Crawler;

var outputDirOption = new Option<string>("--output-dir") { Description = "Dataset directory — gets images.bin/tag_rows.jsonl/tag_vocabulary.json (same format build-large-cache produces) plus crawl.sqlite", Required = true };
var minImagesOption = new Option<int>("--min-images") { Description = "Only crawl tags with at least this many posts on at least one site", DefaultValueFactory = _ => 500 };
var maxImagesOption = new Option<int>("--max-images") { Description = "Target images pulled per eligible tag, combined across all sites — each site still searches until it personally accounts for its own even share (ceil(--max-images / site count)), even after the combined total looks met, so one faster/bigger site can't starve a slower one out of ever contributing; a tag's actual combined count can end up slightly over this target as a result", DefaultValueFactory = _ => 1000 };

var httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
// Danbooru/Gelbooru (Cloudflare-fronted) reject requests with no User-Agent as a 403.
httpClient.DefaultRequestHeaders.UserAgent.ParseAdd("unbooru-tagger-data-booru-downloader/1.0 (+https://github.com/unbooru)");

static Dictionary<string, IBooruClient> BuildClients(
    IReadOnlyList<string> sites,
    HttpClient httpClient,
    double rateDanbooru,
    double rateGelbooru,
    string? danbooruLogin,
    string? danbooruApiKey,
    string? gelbooruApiKey,
    string? gelbooruUserId)
{
    var clients = new Dictionary<string, IBooruClient>(StringComparer.Ordinal);
    foreach (var site in sites)
    {
        IBooruClient client = site switch
        {
            "danbooru" => new DanbooruClient(httpClient, new FixedIntervalRateLimiter(rateDanbooru), danbooruLogin, danbooruApiKey),
            "gelbooru" => new GelbooruClient(httpClient, new FixedIntervalRateLimiter(rateGelbooru), gelbooruApiKey, gelbooruUserId),
            _ => throw new ArgumentException($"Unknown site '{site}'. Expected 'danbooru' or 'gelbooru'.")
        };
        clients[site] = client;
    }
    return clients;
}

/// <summary>
/// Wraps <see cref="TagAliasCache.FetchAndCacheAsync"/> for a command that structurally
/// depends on alias data for correctness — <c>survey-tags</c> to merge a cross-site
/// alias (e.g. <c>head_pat</c>/<c>headpat</c>) into one eligible tag, <c>refresh-tags</c>
/// to reconcile images already stuck on an aliased-away identity. A raw fetch failure
/// (Danbooru unreachable, rate-limited, ...) would otherwise either crash the whole
/// command with an unhandled-exception stack trace, or — worse — silently proceed with
/// <see langword="null"/> aliases and produce a survey/reconciliation that's quietly
/// wrong instead of visibly incomplete. Neither is acceptable for a command whose entire
/// job depends on this data, so this exits cleanly instead: a short, actionable message
/// and a non-zero exit code, no partial/incorrect work performed.
/// </summary>
static async Task<(Dictionary<string, string>? TagAliases, bool Failed)> FetchTagAliasesOrFailAsync(
    string commandName, string outputDirectory, IReadOnlyDictionary<string, IBooruClient> clients, CancellationToken cancellationToken)
{
    try
    {
        return (await TagAliasCache.FetchAndCacheAsync(outputDirectory, clients, cancellationToken), false);
    }
    catch (Exception ex) when (ex is not OperationCanceledException)
    {
        Console.Error.WriteLine($"Failed to fetch active tag aliases: {ex.Message}");
        Console.Error.WriteLine($"'{commandName}' depends on this to correctly merge/reconcile cross-site tag aliases (e.g. head_pat/headpat, or Gelbooru's own curvy_figure/curvy) — exiting without making any changes rather than risk a silently wrong result. Try again once connectivity is restored.");
        return (null, true);
    }
}

static void PrintEstimate(CrawlEstimate estimate)
{
    var table = new Table().Border(TableBorder.Rounded);
    table.AddColumn("Metric");
    table.AddColumn("Value");
    table.AddRow("Eligible tags", estimate.EligibleTagCount.ToString());
    table.AddRow("Estimated image slots (pre-dedup upper bound)", estimate.EstimatedImageSlots.ToString());
    table.AddRow("Estimated requests", estimate.EstimatedRequests.ToString());
    table.AddRow("Estimated wall-clock time (best case)", estimate.EstimatedWallClockTime.ToString(@"d\.hh\:mm\:ss"));
    AnsiConsole.Write(table);
    AnsiConsole.MarkupLine("[grey]Actual unique images will be lower once cross-tag/cross-site dedup applies — this is an upper bound, not a prediction of final corpus size.[/]");
}

var sitesOption = new Option<string[]>("--sites")
{
    Description = "Which site(s) to survey/crawl",
    DefaultValueFactory = _ => ["danbooru", "gelbooru"],
    AllowMultipleArgumentsPerToken = true
};

var danbooruLoginOption = new Option<string?>("--danbooru-login") { Description = "Danbooru username (paired with --danbooru-api-key) for a higher rate-limit tier — also required to download the image file for login-gated posts (e.g. loli/shota-adjacent tags): those posts still show up in search results without it, but come back with no file_url and get silently skipped, so an account with access to that content is needed to actually crawl those tags" };
var danbooruApiKeyOption = new Option<string?>("--danbooru-api-key") { Description = "Danbooru API key" };
var gelbooruApiKeyOption = new Option<string?>("--gelbooru-api-key") { Description = "Gelbooru API key — Gelbooru now requires this (paired with --gelbooru-user-id) even for read-only tag/post listing, or requests fail with 401" };
var gelbooruUserIdOption = new Option<string?>("--gelbooru-user-id") { Description = "Gelbooru user id (paired with --gelbooru-api-key)" };
var rateDanbooruOption = new Option<double>("--rate-danbooru") { Description = "Danbooru requests/second (site's documented global read limit is 10/s)", DefaultValueFactory = _ => 4.0 };
var rateGelbooruOption = new Option<double>("--rate-gelbooru") { Description = "Gelbooru requests/second (undocumented limit — stay conservative)", DefaultValueFactory = _ => 2.0 };

var surveyCommand = new Command("survey-tags", "Survey per-site tag post counts and record eligibility (>= --min-images on at least one site)");
surveyCommand.Options.Add(sitesOption);
surveyCommand.Options.Add(minImagesOption);
surveyCommand.Options.Add(maxImagesOption);
surveyCommand.Options.Add(outputDirOption);
surveyCommand.Options.Add(danbooruLoginOption);
surveyCommand.Options.Add(danbooruApiKeyOption);
surveyCommand.Options.Add(gelbooruApiKeyOption);
surveyCommand.Options.Add(gelbooruUserIdOption);
surveyCommand.Options.Add(rateDanbooruOption);
surveyCommand.Options.Add(rateGelbooruOption);
surveyCommand.SetAction(async (parseResult, cancellationToken) =>
{
    var outputDirectory = parseResult.GetRequiredValue(outputDirOption);
    var minImages = parseResult.GetRequiredValue(minImagesOption);
    var maxImages = parseResult.GetRequiredValue(maxImagesOption);
    var sites = parseResult.GetRequiredValue(sitesOption);
    var rateDanbooru = parseResult.GetRequiredValue(rateDanbooruOption);
    var rateGelbooru = parseResult.GetRequiredValue(rateGelbooruOption);

    using var db = await CrawlDatabase.OpenOrCreateAsync(outputDirectory, cancellationToken);
    var clients = BuildClients(
        sites, httpClient, rateDanbooru, rateGelbooru,
        parseResult.GetValue(danbooruLoginOption), parseResult.GetValue(danbooruApiKeyOption),
        parseResult.GetValue(gelbooruApiKeyOption), parseResult.GetValue(gelbooruUserIdOption));
    var excludedTags = await TagExclusions.LoadOrCreateAsync(outputDirectory, cancellationToken);
    var (tagAliases, tagAliasFetchFailed) = await FetchTagAliasesOrFailAsync("survey-tags", outputDirectory, clients, cancellationToken);
    if (tagAliasFetchFailed)
        return 1;

    var stopNotes = new List<string>();
    var summary = await AnsiConsole.Progress()
        .Columns(ProgressBarColumns.Default)
        .StartAsync(async ctx =>
        {
            var task = ctx.AddTask("Surveying tags...");
            task.IsIndeterminate = true;
            var progress = new TagSurveyProgress
            {
                OnSiteTagCount = (site, count) => task.Description = $"Surveying tags... ({site}: {count} eligible-so-far)",
                OnSiteStopped = (site, eligibleCount, tagName, postCount) =>
                    stopNotes.Add($"{site}: stopped after {eligibleCount} eligible tag(s) — first tag under quota was '{tagName}' ({postCount} < --min-images {minImages})"),
                OnPersisting = (written, total) =>
                {
                    task.IsIndeterminate = false;
                    task.MaxValue = total;
                    task.Value = written;
                    task.Description = $"Persisting survey results to crawl.sqlite... ({written}/{total})";
                }
            };
            return await TagSurveyor.SurveyAsync(db, clients.Values.ToList(), minImages, maxImages, excludedTags, tagAliases, progress, cancellationToken);
        });

    foreach (var note in stopNotes)
        AnsiConsole.MarkupLine($"[grey]{Markup.Escape(note)}[/]");
    if (stopNotes.Any(n => n.Contains("stopped after 0 eligible")))
        AnsiConsole.MarkupLine("[yellow]A site found zero eligible tags — if that's unexpected, the site's tag-list API may not actually be honoring the sort-by-count-descending request, so this stopped on the very first (effectively random) tag instead of the least popular eligible one.[/]");

    Console.WriteLine($"Surveyed {summary.TotalTagsSeen} tags; {summary.EligibleTagCount} eligible at --min-images {minImages} ({summary.ExcludedTagCount} excluded via '{TagExclusions.FileName}', {summary.DeprecatedTagCount} deprecated on at least one site — still crawled, informational only).");
    PrintEstimate(new CrawlEstimate(summary.EligibleTagCount, summary.EstimatedImageSlots, 0, TimeSpan.Zero));
    return 0;
});

var negativeTargetOption = new Option<int?>("--negative-target") { Description = "Non-tagged images each eligible tag should end up with (default 2x --min-images — see plan notes on dedup bias)" };
var inputSizeOption = new Option<int>("--input-size") { Description = "Square input resolution to preprocess images to", DefaultValueFactory = _ => 224 };
var vocabCompactIntervalOption = new Option<int>("--vocab-compact-interval") { Description = "Pages between full tag_vocabulary.json compactions (the delta log is still checkpointed every page regardless, same as build-large-cache's option of the same name)", DefaultValueFactory = _ => 20 };
var negativeCooccurrenceRatioOption = new Option<double>("--negative-cooccurrence-ratio") { Description = "Minimum fraction of a tag's own images that must also carry a candidate tag before that candidate is trusted as a hard-negative source", DefaultValueFactory = _ => 0.5 };
var negativeCooccurrenceMinExamplesOption = new Option<int>("--negative-cooccurrence-min-examples") { Description = "Minimum counter-example images (has the candidate tag, lacks the target) required to trust a pair as a hard-negative source", DefaultValueFactory = _ => 15 };
var maxHardNegativeSourcesOption = new Option<int>("--max-hard-negative-sources") { Description = "Cap on distinct co-occurring tags tried as hard-negative queries per tag before falling back to the plain tag-absent negative query; 0 disables hard-negative mining", DefaultValueFactory = _ => 3 };

var resumeOption = new Option<bool>("--resume")
{
    Description = $"Reuse every crawl option (sites, --min-images/--max-images/--input-size, rates, API credentials, negative-mining settings) from the last 'crawl' run recorded under --output-dir ('{CrawlCommandRecord.FileName}'), ignoring any of those options passed alongside this flag — only --output-dir is needed together with --resume. Fails if no prior 'crawl' run was recorded there yet."
};

var crawlCommand = new Command("crawl", "Download images for every eligible tag (rarest-first), then top up negatives — writes directly into a trainable dataset directory");
crawlCommand.Options.Add(sitesOption);
crawlCommand.Options.Add(minImagesOption);
crawlCommand.Options.Add(maxImagesOption);
crawlCommand.Options.Add(outputDirOption);
crawlCommand.Options.Add(inputSizeOption);
crawlCommand.Options.Add(danbooruLoginOption);
crawlCommand.Options.Add(danbooruApiKeyOption);
crawlCommand.Options.Add(gelbooruApiKeyOption);
crawlCommand.Options.Add(gelbooruUserIdOption);
crawlCommand.Options.Add(rateDanbooruOption);
crawlCommand.Options.Add(rateGelbooruOption);
crawlCommand.Options.Add(negativeTargetOption);
crawlCommand.Options.Add(vocabCompactIntervalOption);
crawlCommand.Options.Add(negativeCooccurrenceRatioOption);
crawlCommand.Options.Add(negativeCooccurrenceMinExamplesOption);
crawlCommand.Options.Add(maxHardNegativeSourcesOption);
crawlCommand.Options.Add(resumeOption);
crawlCommand.SetAction(async (parseResult, cancellationToken) =>
{
    var outputDirectory = parseResult.GetRequiredValue(outputDirOption);
    var resume = parseResult.GetRequiredValue(resumeOption);

    string[] sites;
    int minImages, maxImages, inputSize, negativeTarget, vocabCompactInterval, negativeCooccurrenceMinExamples, maxHardNegativeSources;
    double rateDanbooru, rateGelbooru, negativeCooccurrenceRatio;
    string? danbooruLogin, danbooruApiKey, gelbooruApiKey, gelbooruUserId;

    if (resume)
    {
        var saved = await CrawlCommandRecord.TryLoadAsync(outputDirectory, cancellationToken);
        if (saved is null)
        {
            Console.Error.WriteLine($"--resume was given but no saved crawl command was found under '{outputDirectory}' ('{CrawlCommandRecord.FileName}') — run 'crawl' normally (without --resume) at least once first.");
            return 1;
        }

        Console.WriteLine($"Resuming with the crawl options saved from the last run under '{outputDirectory}'.");
        sites = saved.Sites;
        minImages = saved.MinImages;
        maxImages = saved.MaxImages;
        inputSize = saved.InputSize;
        danbooruLogin = saved.DanbooruLogin;
        danbooruApiKey = saved.DanbooruApiKey;
        gelbooruApiKey = saved.GelbooruApiKey;
        gelbooruUserId = saved.GelbooruUserId;
        rateDanbooru = saved.RateDanbooru;
        rateGelbooru = saved.RateGelbooru;
        negativeTarget = saved.NegativeTarget;
        vocabCompactInterval = saved.VocabCompactInterval;
        negativeCooccurrenceRatio = saved.NegativeCooccurrenceRatio;
        negativeCooccurrenceMinExamples = saved.NegativeCooccurrenceMinExamples;
        maxHardNegativeSources = saved.MaxHardNegativeSources;
    }
    else
    {
        sites = parseResult.GetRequiredValue(sitesOption);
        minImages = parseResult.GetRequiredValue(minImagesOption);
        maxImages = parseResult.GetRequiredValue(maxImagesOption);
        inputSize = parseResult.GetRequiredValue(inputSizeOption);
        rateDanbooru = parseResult.GetRequiredValue(rateDanbooruOption);
        rateGelbooru = parseResult.GetRequiredValue(rateGelbooruOption);
        danbooruLogin = parseResult.GetValue(danbooruLoginOption);
        danbooruApiKey = parseResult.GetValue(danbooruApiKeyOption);
        gelbooruApiKey = parseResult.GetValue(gelbooruApiKeyOption);
        gelbooruUserId = parseResult.GetValue(gelbooruUserIdOption);
        negativeTarget = parseResult.GetValue(negativeTargetOption) ?? minImages * 2;
        vocabCompactInterval = parseResult.GetRequiredValue(vocabCompactIntervalOption);
        negativeCooccurrenceRatio = parseResult.GetRequiredValue(negativeCooccurrenceRatioOption);
        negativeCooccurrenceMinExamples = parseResult.GetRequiredValue(negativeCooccurrenceMinExamplesOption);
        maxHardNegativeSources = parseResult.GetRequiredValue(maxHardNegativeSourcesOption);

        await CrawlCommandRecord.SaveAsync(outputDirectory, new CrawlCommandRecord(
            sites, minImages, maxImages, inputSize, danbooruLogin, danbooruApiKey, gelbooruApiKey, gelbooruUserId,
            rateDanbooru, rateGelbooru, negativeTarget, vocabCompactInterval, negativeCooccurrenceRatio,
            negativeCooccurrenceMinExamples, maxHardNegativeSources), cancellationToken);
    }

    using var db = await CrawlDatabase.OpenOrCreateAsync(outputDirectory, cancellationToken);

    var allTags = await db.GetAllSurveyedTagsAsync(cancellationToken);
    if (allTags.Count == 0)
    {
        Console.Error.WriteLine("No surveyed tags found — run 'survey-tags' against this --output-dir first.");
        return 1;
    }

    var clients = BuildClients(
        sites, httpClient, rateDanbooru, rateGelbooru,
        danbooruLogin, danbooruApiKey, gelbooruApiKey, gelbooruUserId);
    var excludedTags = await TagExclusions.LoadOrCreateAsync(outputDirectory, cancellationToken);
    // Never fetches — only survey-tags/refresh-tags populate this cache; crawl just
    // reads whatever's already there (see TagAliasCache's own doc comment for why).
    var tagAliases = await TagAliasCache.TryLoadAsync(outputDirectory, cancellationToken);

    var pageSizeBySite = clients.ToDictionary(kv => kv.Key, kv => kv.Value.PageSize);
    var rateBySite = new Dictionary<string, double> { ["danbooru"] = rateDanbooru, ["gelbooru"] = rateGelbooru }
        .Where(kv => clients.ContainsKey(kv.Key))
        .ToDictionary(kv => kv.Key, kv => kv.Value);

    var estimate = DatasetCrawler.Estimate(allTags, minImages, maxImages, pageSizeBySite, rateBySite, tagSurveyRequestsMade: 0, excludedTags);
    Console.WriteLine("Estimate (recomputed from the last survey-tags run):");
    PrintEstimate(estimate);

    var result = await AnsiConsole.Progress()
        .Columns(ProgressBarColumns.Default)
        .StartAsync(async ctx =>
        {
            var progress = ProgressBarColumns.AddCrawlTasks(ctx, sites);
            return await DatasetCrawler.RunAsync(
                db, clients, httpClient, outputDirectory, inputSize,
                minImages, maxImages, negativeTarget, vocabCompactInterval,
                progress, cancellationToken, excludedTags, tagAliases: tagAliases,
                negativeCooccurrenceRatio: negativeCooccurrenceRatio,
                negativeCooccurrenceMinExamples: negativeCooccurrenceMinExamples,
                maxHardNegativeSources: maxHardNegativeSources);
        });

    if (result.Shortfalls.Count > 0)
    {
        AnsiConsole.MarkupLine($"[yellow]{result.Shortfalls.Count} eligible tag(s) fell short of --max-images {maxImages} — both sites ran out of posts for these before dedup (md5 + perceptual hash) let the tag reach quota:[/]");
        var shortfallTable = new Table().Border(TableBorder.Rounded);
        shortfallTable.AddColumn("Tag");
        shortfallTable.AddColumn("Achieved");
        shortfallTable.AddColumn("Target");
        foreach (var shortfall in result.Shortfalls)
            shortfallTable.AddRow(shortfall.TagName, shortfall.Achieved.ToString(), shortfall.Target.ToString());
        AnsiConsole.Write(shortfallTable);
    }

    var errorLogPath = CrawlErrorLog.ForDirectory(outputDirectory).LogPath;
    if (File.Exists(errorLogPath))
        AnsiConsole.MarkupLine($"[yellow]One or more sites hit an error during this run (each retried automatically) — see '{Markup.Escape(errorLogPath)}' for a durable record.[/]");

    Console.WriteLine($"Done. Dataset written to '{outputDirectory}'.");
    return 0;
});

var resetOption = new Option<bool>("--reset") { Description = "Restart each selected site's refresh sweep from the beginning instead of resuming after the last post it checked — use to re-verify posts a normal refresh already passed, e.g. after a change to how reconciliation works" };
var onlyTagsOption = new Option<string[]>("--only-tags")
{
    Description = "Skip the full per-site sweep entirely and instead only re-check images that currently hold one of these tag identities — for a scoped correction (e.g. reconciling images stuck on a tag a tag-alias merge just orphaned) where sweeping the whole corpus would be far more work than the problem needs. Never touches --reset/the normal resumable cursor.",
    AllowMultipleArgumentsPerToken = true,
    DefaultValueFactory = _ => []
};

var refreshCommand = new Command("refresh-tags", "Re-fetch previously-crawled posts by id to catch tag edits made on the site since crawl last saw them, reconciling each affected image's tags as the union of every known source's current tags (can both add and remove)");
refreshCommand.Options.Add(sitesOption);
refreshCommand.Options.Add(minImagesOption);
refreshCommand.Options.Add(outputDirOption);
refreshCommand.Options.Add(inputSizeOption);
refreshCommand.Options.Add(danbooruLoginOption);
refreshCommand.Options.Add(danbooruApiKeyOption);
refreshCommand.Options.Add(gelbooruApiKeyOption);
refreshCommand.Options.Add(gelbooruUserIdOption);
refreshCommand.Options.Add(rateDanbooruOption);
refreshCommand.Options.Add(rateGelbooruOption);
refreshCommand.Options.Add(resetOption);
refreshCommand.Options.Add(onlyTagsOption);
refreshCommand.SetAction(async (parseResult, cancellationToken) =>
{
    var outputDirectory = parseResult.GetRequiredValue(outputDirOption);
    var minImages = parseResult.GetRequiredValue(minImagesOption);
    var sites = parseResult.GetRequiredValue(sitesOption);
    var inputSize = parseResult.GetRequiredValue(inputSizeOption);
    var rateDanbooru = parseResult.GetRequiredValue(rateDanbooruOption);
    var rateGelbooru = parseResult.GetRequiredValue(rateGelbooruOption);
    var reset = parseResult.GetRequiredValue(resetOption);
    var onlyTags = parseResult.GetRequiredValue(onlyTagsOption);

    using var db = await CrawlDatabase.OpenOrCreateAsync(outputDirectory, cancellationToken);

    var allTags = await db.GetAllSurveyedTagsAsync(cancellationToken);
    if (allTags.Count == 0)
    {
        Console.Error.WriteLine("No surveyed tags found — run 'survey-tags' (and 'crawl') against this --output-dir first.");
        return 1;
    }

    var clients = BuildClients(
        sites, httpClient, rateDanbooru, rateGelbooru,
        parseResult.GetValue(danbooruLoginOption), parseResult.GetValue(danbooruApiKeyOption),
        parseResult.GetValue(gelbooruApiKeyOption), parseResult.GetValue(gelbooruUserIdOption));
    var excludedTags = await TagExclusions.LoadOrCreateAsync(outputDirectory, cancellationToken);
    var (tagAliases, tagAliasFetchFailed) = await FetchTagAliasesOrFailAsync("refresh-tags", outputDirectory, clients, cancellationToken);
    if (tagAliasFetchFailed)
        return 1;

    RefreshResult result;
    try
    {
        result = await AnsiConsole.Progress()
            .Columns(ProgressBarColumns.Default)
            .StartAsync(async ctx =>
            {
                var progress = ProgressBarColumns.AddRefreshTasks(ctx, sites);
                return await TagRefresher.RunAsync(db, clients, outputDirectory, inputSize, minImages, reset, progress, cancellationToken, excludedTags, tagAliases, onlyTags);
            });
    }
    catch (AllSitesUnavailableException ex)
    {
        AnsiConsole.MarkupLine("[red]Every configured site failed — nothing left to refresh with:[/]");
        foreach (var (site, reason) in ex.FailedSites)
            AnsiConsole.MarkupLine($"[red]  {Markup.Escape(site)}: {Markup.Escape(reason)}[/]");
        Console.Error.WriteLine("Exiting. Refresh progress is checkpointed, so re-running once connectivity is restored resumes without redoing work.");
        return 1;
    }

    if (result.FailedSites.Count > 0)
    {
        AnsiConsole.MarkupLine("[yellow]The following site(s) went unavailable partway through this run and were skipped for its remainder — the next run will pick each back up from where it left off:[/]");
        foreach (var (site, reason) in result.FailedSites)
            AnsiConsole.MarkupLine($"[yellow]  {Markup.Escape(site)}: {Markup.Escape(reason)}[/]");
    }

    Console.WriteLine($"Done. Checked {result.SourcesChecked} source(s); {result.ImagesChanged} image(s) had their tags updated.");
    return 0;
});

var shrinkCacheCommand = new Command("shrink-cache", "One-time conversion of a dataset directory's images.bin from the old float32/full-padded-canvas format to the current uint8/content-only format, in place — re-run to resume after an interruption");
shrinkCacheCommand.Options.Add(outputDirOption);
shrinkCacheCommand.SetAction((parseResult, cancellationToken) =>
{
    var outputDirectory = parseResult.GetRequiredValue(outputDirOption);

    AnsiConsole.Progress()
        .Columns(ProgressBarColumns.Default)
        .Start(ctx =>
        {
            var task = ctx.AddTask("Shrinking cache...");
            task.IsIndeterminate = true;
            PreprocessedDatasetCacheMigrator.ShrinkInPlace(
                outputDirectory,
                (converted, total) =>
                {
                    task.IsIndeterminate = false;
                    task.MaxValue = total;
                    task.Value = converted;
                    task.Description = $"Shrinking cache... ({converted}/{total})";
                },
                cancellationToken);
        });

    Console.WriteLine($"Done. '{outputDirectory}' is now in the current cache format.");
    return Task.FromResult(0);
});

// Shared by every dry-run-by-default correction command (prune-aliased-tags,
// detect-deprecated-redirects, rebuild-vocabulary, prune-unsupported-media), so the
// wording has to stay generic — it used to describe only prune-aliased-tags' own deletes
// and read as plainly wrong on the three commands that don't delete anything.
var applyOption = new Option<bool>("--apply") { Description = "Actually make the changes — without this, only reports what would be done", DefaultValueFactory = _ => false };

var pruneAliasedTagsCommand = new Command(
    "prune-aliased-tags",
    "One-off crawl.sqlite correction: re-fetches Danbooru's and Gelbooru's current tag-alias tables and removes any tag still surveyed as its own eligible tag that one of them has since aliased away to a different canonical spelling (e.g. Gelbooru's own 'curvy_figure' -> 'curvy', or Danbooru's 'nude_male_clothed_female' vs. Gelbooru's own 'clothed_female_nude_male') — without re-running the full 'survey-tags' pass. A tag stuck like this can never earn that site's fairness-floor credit under its own identity: every post the site's search returns for it genuinely carries the ALIAS TARGET's raw tag, not the searched one, so the crawler pages through the site's entire result set for zero counted progress. Also refreshes the on-disk alias cache so 'crawl --resume'/'refresh-tags' pick up the correction. Defaults to a dry run — pass --apply to actually delete.");
pruneAliasedTagsCommand.Options.Add(sitesOption);
pruneAliasedTagsCommand.Options.Add(outputDirOption);
pruneAliasedTagsCommand.Options.Add(danbooruLoginOption);
pruneAliasedTagsCommand.Options.Add(danbooruApiKeyOption);
pruneAliasedTagsCommand.Options.Add(gelbooruApiKeyOption);
pruneAliasedTagsCommand.Options.Add(gelbooruUserIdOption);
pruneAliasedTagsCommand.Options.Add(rateDanbooruOption);
pruneAliasedTagsCommand.Options.Add(rateGelbooruOption);
pruneAliasedTagsCommand.Options.Add(applyOption);
pruneAliasedTagsCommand.SetAction(async (parseResult, cancellationToken) =>
{
    var outputDirectory = parseResult.GetRequiredValue(outputDirOption);
    var sites = parseResult.GetRequiredValue(sitesOption);
    var rateDanbooru = parseResult.GetRequiredValue(rateDanbooruOption);
    var rateGelbooru = parseResult.GetRequiredValue(rateGelbooruOption);
    var apply = parseResult.GetRequiredValue(applyOption);

    using var db = await CrawlDatabase.OpenOrCreateAsync(outputDirectory, cancellationToken);

    var allTags = await db.GetAllSurveyedTagsAsync(cancellationToken);
    if (allTags.Count == 0)
    {
        Console.Error.WriteLine("No surveyed tags found — run 'survey-tags' against this --output-dir first.");
        return 1;
    }

    var clients = BuildClients(
        sites, httpClient, rateDanbooru, rateGelbooru,
        parseResult.GetValue(danbooruLoginOption), parseResult.GetValue(danbooruApiKeyOption),
        parseResult.GetValue(gelbooruApiKeyOption), parseResult.GetValue(gelbooruUserIdOption));
    var (tagAliases, tagAliasFetchFailed) = await FetchTagAliasesOrFailAsync("prune-aliased-tags", outputDirectory, clients, cancellationToken);
    if (tagAliasFetchFailed)
        return 1;
    if (tagAliases is not { Count: > 0 })
    {
        Console.WriteLine("No active tag aliases found on any configured site — nothing to prune.");
        return 0;
    }

    // Same test TagSurveyor.SurveyAsync uses to identify a stale antecedent row: a
    // currently-surveyed tag whose own raw name is now a known alias antecedent.
    var staleRows = allTags.Where(t => tagAliases.ContainsKey(TagCategoryNaming.RawName(t.Name))).ToList();
    if (staleRows.Count == 0)
    {
        Console.WriteLine($"Checked {allTags.Count} surveyed tag(s) against {tagAliases.Count} active alias(es) across {clients.Count} site(s) — none of them are a known alias antecedent. Nothing to prune.");
        return 0;
    }

    var reportTable = new Table().Border(TableBorder.Rounded);
    reportTable.AddColumn("Currently its own tag");
    reportTable.AddColumn("Actually an alias of");
    reportTable.AddColumn("Danbooru count");
    reportTable.AddColumn("Gelbooru count");
    foreach (var tag in staleRows)
    {
        var target = tagAliases[TagCategoryNaming.RawName(tag.Name)];
        reportTable.AddRow(tag.Name, target, tag.DanbooruCount?.ToString() ?? "-", tag.GelbooruCount?.ToString() ?? "-");
    }
    AnsiConsole.Write(reportTable);

    if (!apply)
    {
        Console.WriteLine($"{staleRows.Count} tag(s) would be removed from Tags/TagProgress (dry run — pass --apply to actually delete them).");
        return 0;
    }

    var names = staleRows.Select(t => t.Name).ToList();
    await db.DeleteTagSurveysAsync(names, cancellationToken);
    await db.DeleteTagProgressAsync(names, cancellationToken);

    Console.WriteLine($"Removed {names.Count} tag(s) from Tags/TagProgress — 'crawl' will no longer iterate them as their own tag.");
    Console.WriteLine($"If any already have downloaded images tagged under the old identity, reconcile them onto their alias target with:");
    Console.WriteLine($"  refresh-tags --output-dir {outputDirectory} --only-tags {string.Join(' ', names)}");
    Console.WriteLine("A 'crawl' run re-reads the full tag list from crawl.sqlite on every invocation (including --resume), so restarting it — no special flag needed — is enough to pick up this correction.");
    return 0;
});

var redirectLimitOption = new Option<int>("--limit") { Description = "Only probe the first N deprecated tags (highest post count first, so a partial run covers the most valuable ones) — every tag needs at least two live searches, so a full corpus-wide run is thousands of rate-limited round-trips; 0 means no limit", DefaultValueFactory = _ => 0 };
var redirectSamplePagesOption = new Option<int>("--sample-pages") { Description = "Pages of search results sampled per tag and per candidate (~200 posts per page-pair on either site). Bigger samples make an exact match stronger evidence and shrink the candidate set, at proportionally more requests — though a candidate that already differs on page 1 is rejected without fetching the rest", DefaultValueFactory = _ => 2 };
var redirectMinSampleOption = new Option<int>("--min-sample-posts") { Description = "Leave a tag unresolved rather than trust a match, if the site returned fewer than this many posts for it — an exact match over a handful of posts is cheap coincidence, not evidence", DefaultValueFactory = _ => 20 };
var redirectMaxCandidatesOption = new Option<int>("--max-candidates") { Description = "Skip a tag entirely (rather than verify an arbitrary subset) if more than this many tags are present on 100% of its sampled posts — a candidate set that big means the sample was too small to discriminate, and the fix is a larger --sample-pages", DefaultValueFactory = _ => 8 };
var redirectMaxAntecedentsOption = new Option<int>("--max-antecedents-per-target") { Description = "Distrust and write NOTHING for every confirmation pointing at a target that more than this many distinct tags converged on in the same run — the signature of a site-wide search FALLBACK, not a redirect (Gelbooru answers a query it can't parse, e.g. 'l' or '1:1', with a shared default result set identical to a real search for 'bad_tag', so every such query verifies against it with a genuinely exact match). A real target absorbing dozens of typo/synonym variants is normal; hundreds of unrelated ones is not. The default sits ~3x above the largest real cluster seen so far (47) and ~7x below the observed artifact (1000)", DefaultValueFactory = _ => 150 };
var redirectOnlyTagsOption = new Option<string[]>("--only-tags")
{
    Description = "Probe only these tag identities instead of every deprecated+eligible tag — for a scoped re-verification (e.g. re-checking just the tags a previous run got wrong) where re-probing the whole corpus would be thousands of needless rate-limited round-trips. Takes the same stored identities this command's own report prints. Applied before --limit.",
    AllowMultipleArgumentsPerToken = true,
    DefaultValueFactory = _ => []
};

var detectDeprecatedRedirectsCommand = new Command(
    "detect-deprecated-redirects",
    "Finds tags a site has DEPRECATED and now silently redirects searches for, with zero trace in any alias API — the gap 'prune-aliased-tags' can't cover, since that one only knows what Danbooru/Gelbooru actually publish as aliases. Confirmed live: Gelbooru's 'pussy_licking' (deprecated via its own type=6) returns the identical post-id sequence as 'cunnilingus' while appearing in no alias listing anywhere. Detection is empirical and per-tag: sample the deprecated tag's own search, take the tags present on 100% of those posts as candidates, then independently re-search each candidate and require an exact, ordered post-id match — co-occurrence alone proves nothing, only the exact match proves the two are the same underlying query. Confirmed pairs are MERGED into tag_aliases.json (never overwriting an official alias already there) and recorded separately in detected_tag_redirects.json so a later survey-tags run folds them back in instead of discarding them. Defaults to a dry run — pass --apply to actually write.");
detectDeprecatedRedirectsCommand.Options.Add(sitesOption);
detectDeprecatedRedirectsCommand.Options.Add(outputDirOption);
detectDeprecatedRedirectsCommand.Options.Add(minImagesOption);
detectDeprecatedRedirectsCommand.Options.Add(danbooruLoginOption);
detectDeprecatedRedirectsCommand.Options.Add(danbooruApiKeyOption);
detectDeprecatedRedirectsCommand.Options.Add(gelbooruApiKeyOption);
detectDeprecatedRedirectsCommand.Options.Add(gelbooruUserIdOption);
detectDeprecatedRedirectsCommand.Options.Add(rateDanbooruOption);
detectDeprecatedRedirectsCommand.Options.Add(rateGelbooruOption);
detectDeprecatedRedirectsCommand.Options.Add(redirectLimitOption);
detectDeprecatedRedirectsCommand.Options.Add(redirectSamplePagesOption);
detectDeprecatedRedirectsCommand.Options.Add(redirectMinSampleOption);
detectDeprecatedRedirectsCommand.Options.Add(redirectMaxCandidatesOption);
detectDeprecatedRedirectsCommand.Options.Add(redirectMaxAntecedentsOption);
detectDeprecatedRedirectsCommand.Options.Add(redirectOnlyTagsOption);
detectDeprecatedRedirectsCommand.Options.Add(applyOption);
detectDeprecatedRedirectsCommand.SetAction(async (parseResult, cancellationToken) =>
{
    var outputDirectory = parseResult.GetRequiredValue(outputDirOption);
    var sites = parseResult.GetRequiredValue(sitesOption);
    var rateDanbooru = parseResult.GetRequiredValue(rateDanbooruOption);
    var rateGelbooru = parseResult.GetRequiredValue(rateGelbooruOption);
    var limit = parseResult.GetRequiredValue(redirectLimitOption);
    var onlyTags = parseResult.GetRequiredValue(redirectOnlyTagsOption);
    var apply = parseResult.GetRequiredValue(applyOption);
    var options = new DeprecatedRedirectOptions
    {
        SamplePages = parseResult.GetRequiredValue(redirectSamplePagesOption),
        MinSamplePosts = parseResult.GetRequiredValue(redirectMinSampleOption),
        MaxCandidates = parseResult.GetRequiredValue(redirectMaxCandidatesOption),
        MaxAntecedentsPerTarget = parseResult.GetRequiredValue(redirectMaxAntecedentsOption),
    };

    using var db = await CrawlDatabase.OpenOrCreateAsync(outputDirectory, cancellationToken);

    var allTags = await db.GetAllSurveyedTagsAsync(cancellationToken);
    if (allTags.Count == 0)
    {
        Console.Error.WriteLine("No surveyed tags found — run 'survey-tags' against this --output-dir first.");
        return 1;
    }

    // Same source of truth for eligibility as rebuild-vocabulary: the crawl's own recorded
    // --min-images, since eligibility is always recomputed live from it (plus
    // excluded_tags.txt) rather than read back out of crawl.sqlite's stored Eligible column.
    var savedCommand = await CrawlCommandRecord.TryLoadAsync(outputDirectory, cancellationToken);
    int minImages;
    if (savedCommand is not null)
    {
        minImages = savedCommand.MinImages;
        Console.WriteLine($"Using --min-images {minImages} from '{CrawlCommandRecord.FileName}' (the value the original crawl actually used).");
    }
    else
    {
        minImages = parseResult.GetRequiredValue(minImagesOption);
        Console.WriteLine($"[warning] No '{CrawlCommandRecord.FileName}' found — using --min-images {minImages}. If that doesn't match the value the original crawl used, which tags count as eligible (and therefore get probed) will be wrong.");
    }

    var clients = BuildClients(
        sites, httpClient, rateDanbooru, rateGelbooru,
        parseResult.GetValue(danbooruLoginOption), parseResult.GetValue(danbooruApiKeyOption),
        parseResult.GetValue(gelbooruApiKeyOption), parseResult.GetValue(gelbooruUserIdOption));
    var excludedTags = await TagExclusions.LoadOrCreateAsync(outputDirectory, cancellationToken);
    // Refreshed rather than read off disk on purpose: an antecedent a site has since
    // aliased officially must be skipped, both to avoid spending two live searches
    // re-deriving something already published and to avoid writing a second, possibly
    // contradicting mapping for it.
    var (tagAliases, tagAliasFetchFailed) = await FetchTagAliasesOrFailAsync("detect-deprecated-redirects", outputDirectory, clients, cancellationToken);
    if (tagAliasFetchFailed)
        return 1;

    var deprecatedTags = allTags
        .Where(t => t.Deprecated && TagEligibility.IsEligible(t, minImages, excludedTags))
        .OrderByDescending(t => t.BestCount)
        .ThenBy(t => t.Name, StringComparer.Ordinal)
        .ToList();
    if (deprecatedTags.Count == 0)
    {
        Console.WriteLine($"None of the {allTags.Count} surveyed tag(s) are both currently eligible and flagged deprecated — nothing to probe.");
        return 0;
    }

    if (onlyTags.Length > 0)
    {
        // Matches stored tag IDENTITIES (category-prefixed for anything but General),
        // exactly like refresh-tags' own --only-tags and exactly what this command's
        // confirmed/suppressed tables print — so a name copied out of a previous report
        // pastes straight back in here.
        var wanted = new HashSet<string>(onlyTags, StringComparer.Ordinal);
        var scoped = deprecatedTags.Where(t => wanted.Contains(t.Name)).ToList();

        var unmatched = wanted.Except(scoped.Select(t => t.Name), StringComparer.Ordinal).ToList();
        if (unmatched.Count > 0)
            AnsiConsole.MarkupLine($"[yellow]{unmatched.Count} of the {wanted.Count} --only-tags name(s) are not currently deprecated+eligible surveyed tags and were skipped (identities, not raw names — e.g. '{Markup.Escape(unmatched[0])}').[/]");

        if (scoped.Count == 0)
        {
            Console.Error.WriteLine("--only-tags matched none of the deprecated+eligible tags — nothing to probe.");
            return 1;
        }

        Console.WriteLine($"--only-tags: scoped from {deprecatedTags.Count} deprecated+eligible tag(s) down to {scoped.Count}.");
        deprecatedTags = scoped;
    }

    var totalDeprecated = deprecatedTags.Count;
    if (limit > 0 && limit < deprecatedTags.Count)
        deprecatedTags = deprecatedTags.Take(limit).ToList();
    Console.WriteLine($"Probing {deprecatedTags.Count} of {totalDeprecated} deprecated+eligible tag(s) across {clients.Count} site(s), against {tagAliases?.Count ?? 0} already-known alias(es).");

    var result = await AnsiConsole.Progress()
        .Columns(ProgressBarColumns.Default)
        .StartAsync(async ctx =>
        {
            var task = ctx.AddTask("Probing deprecated tags...");
            task.MaxValue = deprecatedTags.Count;
            var confirmed = 0;
            var progress = new DeprecatedRedirectProgress
            {
                OnTagStarted = (done, total, tagName) =>
                {
                    task.MaxValue = total;
                    task.Value = done;
                    task.Description = Markup.Escape($"Probing '{tagName}'... ({done}/{total}, {confirmed} redirect(s) confirmed)");
                },
                OnFinding = finding =>
                {
                    if (finding.Outcome == DeprecatedRedirectOutcome.Confirmed)
                        confirmed++;
                }
            };
            var detection = await DeprecatedRedirectDetector.DetectAsync(
                deprecatedTags, clients, tagAliases, options, progress, cancellationToken);
            task.Value = task.MaxValue;
            return detection;
        });

    var outcomeTable = new Table().Border(TableBorder.Rounded);
    outcomeTable.AddColumn("Outcome");
    outcomeTable.AddColumn("Probes");
    foreach (var group in result.Findings.GroupBy(f => f.Outcome).OrderByDescending(g => g.Count()))
        outcomeTable.AddRow(Markup.Escape(group.Key.ToString()), group.Count().ToString());
    outcomeTable.AddRow("[grey]Search requests made[/]", result.RequestsMade.ToString());
    AnsiConsole.Write(outcomeTable);

    var confirmedFindings = result.Findings.Where(f => f.Outcome == DeprecatedRedirectOutcome.Confirmed).ToList();
    if (confirmedFindings.Count > 0)
    {
        var reportTable = new Table().Border(TableBorder.Rounded);
        reportTable.AddColumn("Deprecated tag");
        reportTable.AddColumn("Silently redirects to");
        reportTable.AddColumn("Site");
        reportTable.AddColumn("Posts compared");
        reportTable.AddColumn("Candidates");
        foreach (var finding in confirmedFindings)
            reportTable.AddRow(
                Markup.Escape(finding.TagName),
                Markup.Escape(finding.Target!),
                Markup.Escape(finding.SiteName ?? "-"),
                finding.SampledPosts.ToString(),
                finding.Candidates.Count.ToString());
        AnsiConsole.Write(reportTable);
    }

    var needsAHuman = result.Findings
        .Where(f => f.Outcome is DeprecatedRedirectOutcome.AmbiguousCandidates or DeprecatedRedirectOutcome.WouldCreateCycle)
        .ToList();
    if (needsAHuman.Count > 0)
    {
        AnsiConsole.MarkupLine("[yellow]The following tag(s) produced a match this command deliberately refuses to act on by itself — nothing was written for them:[/]");
        var ambiguousTable = new Table().Border(TableBorder.Rounded);
        ambiguousTable.AddColumn("Deprecated tag");
        ambiguousTable.AddColumn("Site");
        ambiguousTable.AddColumn("Why");
        ambiguousTable.AddColumn("Exact-matching tag(s)");
        foreach (var finding in needsAHuman)
            ambiguousTable.AddRow(
                Markup.Escape(finding.TagName),
                Markup.Escape(finding.SiteName ?? "-"),
                Markup.Escape(finding.Outcome == DeprecatedRedirectOutcome.AmbiguousCandidates
                    ? "several tags reproduce its results exactly — which one the site redirects to is undecidable from searches alone"
                    : "would close an alias cycle against the aliases already known"),
                Markup.Escape(string.Join(", ", finding.VerifiedTargets)));
        AnsiConsole.Write(ambiguousTable);
    }

    // Deliberately its own table rather than another row class in needsAHuman above: this is
    // a RUN-level verdict on a target, not a per-tag one, and a single fallback hub produced
    // a thousand of these in the first real run — folding them in would bury the handful of
    // genuinely per-tag rows a human can actually act on. One row per target, with the
    // antecedent count that IS the evidence, says everything the per-tag rows would.
    var hubFindings = result.Findings
        .Where(f => f.Outcome == DeprecatedRedirectOutcome.SuspectedHubTarget)
        .ToList();
    if (hubFindings.Count > 0)
    {
        var hubGroups = hubFindings
            .GroupBy(f => f.Target!, StringComparer.Ordinal)
            .OrderByDescending(g => g.Count())
            .ToList();
        AnsiConsole.MarkupLine($"[yellow]{hubFindings.Count} confirmation(s) across {hubGroups.Count} target(s) were HELD BACK and not written: more than --max-antecedents-per-target ({options.MaxAntecedentsPerTarget}) distinct tags verified against the same target in this one run. Each match really was exact — which is exactly the problem: a site that answers an unparseable query with a shared fallback result set makes every such query match whatever real tag returns that same set (Gelbooru's 'bad_tag'). Raise the option if one of these is a genuine tag that really does absorb this many variants.[/]");
        var hubTable = new Table().Border(TableBorder.Rounded);
        hubTable.AddColumn("Suspected hub target");
        hubTable.AddColumn("Distinct tags that 'confirmed' against it");
        hubTable.AddColumn("Examples");
        foreach (var group in hubGroups)
            hubTable.AddRow(
                Markup.Escape(group.Key),
                group.Count().ToString(),
                Markup.Escape(string.Join(", ", group.Take(5).Select(f => f.TagName)) + (group.Count() > 5 ? ", ..." : "")));
        AnsiConsole.Write(hubTable);
    }

    if (result.NewAliases.Count == 0)
    {
        Console.WriteLine(hubFindings.Count > 0
            ? "No silent redirects left to write — every confirmation this run made was held back as a suspected hub target (see above). A tag left unresolved is the expected outcome, not an error."
            : "No silent redirects confirmed — nothing to write. A tag left unresolved is the expected outcome, not an error.");
        return 0;
    }

    if (!apply)
    {
        Console.WriteLine($"{result.NewAliases.Count} redirect(s) would be merged into '{TagAliasCache.FileName}' and recorded in '{TagAliasCache.DetectedRedirectsFileName}' (dry run — pass --apply to actually write).");
        return 0;
    }

    var detectedRedirects = new Dictionary<string, string>(
        await TagAliasCache.TryLoadDetectedRedirectsAsync(outputDirectory, cancellationToken) ?? [], StringComparer.Ordinal);
    foreach (var (antecedent, consequent) in result.NewAliases)
        detectedRedirects[antecedent] = consequent;
    await TagAliasCache.SaveDetectedRedirectsAsync(outputDirectory, detectedRedirects, cancellationToken);

    var added = await TagAliasCache.MergeIntoCacheAsync(outputDirectory, result.NewAliases, cancellationToken);
    Console.WriteLine($"Merged {added.Count} new alias(es) into '{Path.Combine(outputDirectory, TagAliasCache.FileName)}' ({detectedRedirects.Count} detected redirect(s) recorded in '{TagAliasCache.DetectedRedirectsFileName}' in total).");
    Console.WriteLine("These tags are now known alias antecedents, so:");
    Console.WriteLine($"  prune-aliased-tags --output-dir {outputDirectory} --apply   # stop crawling them as their own tag");
    // --only-tags matches stored tag IDENTITIES (category-prefixed for anything but
    // General), not the raw names the alias table is keyed by — see TagCategoryNaming.
    Console.WriteLine($"  refresh-tags --output-dir {outputDirectory} --only-tags {string.Join(' ', confirmedFindings.Select(f => f.TagName))}   # move any images already tagged under the old identity");
    return 0;
});

var seedVocabularyOption = new Option<string>("--seed-vocabulary")
{
    Description = "Path to a known-good tag_vocabulary.json to rebuild from -- e.g. a training checkpoint's own copy (Checkpoint.Save writes one byte-for-byte identical to the dataset's vocabulary at that moment). Every tag it already contains keeps its exact row index (row indices are append-only everywhere in this codebase, so an older snapshot is always a verified-correct prefix of a newer one) -- required if a checkpoint trained against this dataset exists, since renumbering those rows would silently invalidate its tag_embeddings.bin.",
    Required = true
};

var rebuildVocabularyCommand = new Command(
    "rebuild-vocabulary",
    "Disaster recovery for a lost/corrupted tag_vocabulary.json: re-derives the tag<->row-index mapping and tag_rows.sqlite from crawl.sqlite's own ImageSources.Tags snapshots (the same ground truth 'refresh-tags' reconciles against), seeded from --seed-vocabulary so any tag it already knows keeps its exact row index. Defaults to a dry run that only reports what it would do -- pass --apply to actually back up and overwrite tag_rows.sqlite/tag_vocabulary.json.");
rebuildVocabularyCommand.Options.Add(outputDirOption);
rebuildVocabularyCommand.Options.Add(seedVocabularyOption);
rebuildVocabularyCommand.Options.Add(minImagesOption);
rebuildVocabularyCommand.Options.Add(applyOption);
rebuildVocabularyCommand.SetAction(async (parseResult, cancellationToken) =>
{
    var outputDirectory = parseResult.GetRequiredValue(outputDirOption);
    var seedVocabularyPath = parseResult.GetRequiredValue(seedVocabularyOption);
    var apply = parseResult.GetRequiredValue(applyOption);

    var savedCommand = await CrawlCommandRecord.TryLoadAsync(outputDirectory, cancellationToken);
    int minImages;
    if (savedCommand is not null)
    {
        minImages = savedCommand.MinImages;
        Console.WriteLine($"Using --min-images {minImages} from '{CrawlCommandRecord.FileName}' (the value the original crawl actually used).");
    }
    else
    {
        minImages = parseResult.GetRequiredValue(minImagesOption);
        Console.WriteLine($"[warning] No '{CrawlCommandRecord.FileName}' found -- using --min-images {minImages}. If this doesn't match the value the original crawl used, eligible-tag membership (and therefore which tags get recovered) will be wrong.");
    }

    var pixelHeaderPath = Path.Combine(outputDirectory, "images.bin");
    int confirmedImageCount;
    await using (var pixelStream = File.OpenRead(pixelHeaderPath))
    {
        using var reader = new BinaryReader(pixelStream);
        reader.ReadInt32(); // format magic -- not re-validated here, images.bin itself is untouched by this command
        confirmedImageCount = reader.ReadInt32();
    }

    var seedVocabulary = TagVocabulary.Load(seedVocabularyPath);
    Console.WriteLine($"Seed vocabulary: {seedVocabulary.Records.Count} tag(s) from '{seedVocabularyPath}'.");

    using var db = await CrawlDatabase.OpenOrCreateAsync(outputDirectory, cancellationToken);
    var excludedTags = await TagExclusions.LoadOrCreateAsync(outputDirectory, cancellationToken);
    var tagAliases = await TagAliasCache.TryLoadAsync(outputDirectory, cancellationToken);

    var (rows, result) = await VocabularyRebuilder.Rebuild(
        db, seedVocabulary, minImages, excludedTags, tagAliases, confirmedImageCount,
        onPhase: phase => Console.WriteLine(phase));

    var reportTable = new Table().Border(TableBorder.Rounded);
    reportTable.AddColumn("Metric");
    reportTable.AddColumn("Value");
    reportTable.AddRow("Seed tag count", result.SeedTagCount.ToString());
    reportTable.AddRow("Final tag count", result.FinalTagCount.ToString());
    reportTable.AddRow("Newly appended tags", (result.FinalTagCount - result.SeedTagCount).ToString());
    reportTable.AddRow("Confirmed images (images.bin)", result.ConfirmedImageCount.ToString());
    reportTable.AddRow("Images recovered from crawl.sqlite", result.ImagesWithSourceData.ToString());
    reportTable.AddRow("Images with no recoverable tags", result.ImagesWithoutSourceData.ToString());
    AnsiConsole.Write(reportTable);

    if (!apply)
    {
        Console.WriteLine("Dry run -- nothing was written. Pass --apply to back up and overwrite tag_rows.sqlite/tag_vocabulary.json.");
        return 0;
    }

    var tagRowsPath = Path.Combine(outputDirectory, "tag_rows.sqlite");
    var tagRowsBackupPath = tagRowsPath + ".pre-rebuild-backup";
    if (File.Exists(tagRowsPath) && !File.Exists(tagRowsBackupPath))
    {
        Console.WriteLine($"Backing up '{tagRowsPath}' to '{tagRowsBackupPath}'...");
        File.Copy(tagRowsPath, tagRowsBackupPath);
    }

    // tag_vocabulary.json is overwritten below by seedVocabulary.Save -- back it up first
    // regardless of whether --seed-vocabulary pointed at this same live file or a separate
    // (e.g. checkpoint) copy, the same way tag_rows.sqlite already is just above. This file
    // is the one that actually mattered in a real incident: rebuild-vocabulary --apply used
    // to overwrite it with no backup at all.
    var vocabularyPath = Path.Combine(outputDirectory, "tag_vocabulary.json");
    var vocabularyBackupPath = vocabularyPath + ".pre-rebuild-backup";
    if (File.Exists(vocabularyPath) && !File.Exists(vocabularyBackupPath))
    {
        Console.WriteLine($"Backing up '{vocabularyPath}' to '{vocabularyBackupPath}'...");
        File.Copy(vocabularyPath, vocabularyBackupPath);
    }

    Console.WriteLine("Writing tag_rows.sqlite...");
    TagRowsSqliteWriter.WriteAtomically(outputDirectory, rows);

    Console.WriteLine("Writing tag_vocabulary.json...");
    seedVocabulary.Save(vocabularyPath);

    var deltaPath = Path.Combine(outputDirectory, "tag_vocabulary.delta.jsonl");
    if (File.Exists(deltaPath))
        File.Delete(deltaPath); // superseded -- the fresh snapshot above already reflects everything a delta would have replayed

    Console.WriteLine("Done.");
    return 0;
});

var pruneUnsupportedMediaCommand = new Command(
    "prune-unsupported-media",
    "One-off correction for images that never should have entered the dataset: 'crawl' now skips video (.webm/.mp4/.mov/.avi/.swf) and .gif URLs before ever downloading them (see UnsupportedMedia.cs -- video always fails to decode as a still image, but GIF is worse: SkiaSharp silently decodes just its first frame instead of failing, so it reads as real complete artwork with nothing marking it as wrong), but this only prevents NEW ones -- it does nothing about ones already sitting in an existing corpus from before that skip existed. This command finds every image whose only recorded source is one of those URLs and removes it entirely (crawl.sqlite rows AND its images.bin/tag_rows.sqlite pixel-cache row), strips just the bad source row off an image that also has a real source, and -- while it's already paying for a full cache rewrite -- also sweeps up pixel-cache rows with no crawl.sqlite tracking at all (e.g. left behind by a past crash that lost the tracking row without the pixel data ever being cleaned up alongside it). Defaults to a dry run -- pass --apply to actually rewrite images.bin/tag_rows.sqlite (a full streaming copy of the whole cache -- can take hours on a large corpus) and crawl.sqlite. Run rebuild-vocabulary afterward to reconcile tag_vocabulary.json's per-tag counts against the now-smaller corpus.");
pruneUnsupportedMediaCommand.Options.Add(outputDirOption);
pruneUnsupportedMediaCommand.Options.Add(applyOption);
pruneUnsupportedMediaCommand.SetAction(async (parseResult, cancellationToken) =>
{
    var outputDirectory = parseResult.GetRequiredValue(outputDirOption);
    var apply = parseResult.GetRequiredValue(applyOption);

    var pixelHeaderPath = Path.Combine(outputDirectory, "images.bin");
    int confirmedImageCount;
    await using (var pixelStream = File.OpenRead(pixelHeaderPath))
    {
        using var reader = new BinaryReader(pixelStream);
        reader.ReadInt32(); // format magic -- not re-validated here, images.bin itself is untouched by this read
        confirmedImageCount = reader.ReadInt32();
    }

    using var db = await CrawlDatabase.OpenOrCreateAsync(outputDirectory, cancellationToken);

    Console.WriteLine("Scanning ImageSources for video/GIF-sourced images...");
    var groups = await db.GetUnsupportedMediaGroupsAsync(cancellationToken);
    var soleSourceGroups = groups.Where(g => g.UnsupportedSourceCount == g.TotalSourceCount).ToList();
    var mixedSourceGroups = groups.Where(g => g.UnsupportedSourceCount < g.TotalSourceCount).ToList();
    var soleSourceMd5s = soleSourceGroups.Select(g => g.Md5).ToHashSet(StringComparer.Ordinal);

    var badRows = await db.GetUnsupportedMediaSourceRowsAsync(cancellationToken);
    var extraSourcesToStrip = badRows
        .Where(r => !soleSourceMd5s.Contains(r.Md5))
        .Select(r => (r.Site, r.PostId))
        .ToList();

    Console.WriteLine("Scanning for pixel-cache rows with no crawl.sqlite tracking...");
    var trackedCacheRows = await db.GetAllCacheRowIndicesAsync(cancellationToken);
    var orphanedCacheRows = Enumerable.Range(0, confirmedImageCount).Where(i => !trackedCacheRows.Contains(i)).ToList();

    var rowsToDrop = soleSourceGroups.Select(g => g.CacheRowIndex).Concat(orphanedCacheRows).ToHashSet();

    var reportTable = new Table().Border(TableBorder.Rounded);
    reportTable.AddColumn("Finding");
    reportTable.AddColumn("Count");
    reportTable.AddRow("Images sourced ONLY from video/GIF URLs (fully removed)", soleSourceGroups.Count.ToString());
    reportTable.AddRow("Images with a mix of sources (bad source row(s) stripped, image kept)", mixedSourceGroups.Count.ToString());
    reportTable.AddRow("  ...of which, bad source rows to strip", extraSourcesToStrip.Count.ToString());
    reportTable.AddRow("Untracked pixel-cache rows (no crawl.sqlite row at all)", orphanedCacheRows.Count.ToString());
    reportTable.AddRow("Total pixel-cache rows to drop", rowsToDrop.Count.ToString());
    reportTable.AddRow("Pixel-cache rows before/after", $"{confirmedImageCount} / {confirmedImageCount - rowsToDrop.Count}");
    AnsiConsole.Write(reportTable);

    if (rowsToDrop.Count == 0 && extraSourcesToStrip.Count == 0)
    {
        Console.WriteLine("Nothing to prune.");
        return 0;
    }

    if (!apply)
    {
        Console.WriteLine("Dry run -- nothing was changed. Pass --apply to actually rewrite images.bin/tag_rows.sqlite/crawl.sqlite.");
        return 0;
    }

    var oldToNewCacheRowIndex = (IReadOnlyDictionary<int, int>)new Dictionary<int, int>();
    if (rowsToDrop.Count > 0)
    {
        Console.WriteLine($"Compacting the pixel cache -- dropping {rowsToDrop.Count} of {confirmedImageCount} rows. This rewrites the entire cache and can take a long time on a large corpus.");
        var compactionResult = AnsiConsole.Progress()
            .Columns(ProgressBarColumns.Default)
            .Start(ctx =>
            {
                var task = ctx.AddTask("Compacting images.bin/tag_rows.sqlite...");
                task.MaxValue = confirmedImageCount;
                return PreprocessedDatasetCacheCompactor.DropRows(
                    outputDirectory,
                    rowsToDrop,
                    onProgress: (processed, total) =>
                    {
                        task.MaxValue = total;
                        task.Value = processed;
                    },
                    cancellationToken);
            });
        oldToNewCacheRowIndex = compactionResult.OldToNewIndex;
        Console.WriteLine($"Cache compacted: {compactionResult.OriginalCount} -> {compactionResult.KeptCount} rows.");
    }

    Console.WriteLine("Applying crawl.sqlite changes...");
    await db.ApplyUnsupportedMediaPruneAsync(soleSourceMd5s, extraSourcesToStrip, oldToNewCacheRowIndex, cancellationToken);

    Console.WriteLine("Done. Run rebuild-vocabulary against this --output-dir to reconcile tag_vocabulary.json's per-tag counts with the now-smaller corpus.");
    return 0;
});

var stagingDirOption = new Option<string>("--staging-dir")
{
    Description = "Fresh directory to build the recovered pixel cache into -- must not be --output-dir. Never touched by anything except this command; left exactly as-is between runs until --finalize succeeds, so it's always safe to interrupt and re-run.",
    Required = true
};
var recoveryConcurrencyOption = new Option<int>("--concurrency") { Description = "How many images to download+decode+encode at once during the slow, network-bound re-download phase", DefaultValueFactory = _ => 8 };
var finalizeOption = new Option<bool>("--finalize")
{
    Description = "Only after the staging cache is fully built and verified: swap it into --output-dir (the damaged images.bin/tag_rows.sqlite there are renamed to '*.broken-pre-recovery.bak', never deleted automatically) and reconcile crawl.sqlite to match. Without this flag, only builds/resumes the staging cache and reports status -- never touches --output-dir beyond reading crawl.sqlite.",
    DefaultValueFactory = _ => false
};

var recoverCorruptedImagesCommand = new Command(
    "recover-corrupted-images",
    "Disaster recovery for a damaged images.bin/tag_rows.sqlite -- e.g. a prune-unsupported-media compaction that reported success at the right final row count while actually writing empty 0x0 records for part of the corpus (a real incident this command exists to recover from; see PreprocessedDatasetCacheCompactor's own doc comments for the write-side verification gap that let it happen). Rebuilds a fresh, independently-verified pixel cache in --staging-dir: copies whatever's still genuinely valid at the front of the damaged cache (fast, local, read-only against --output-dir), then re-downloads everything past that point from crawl.sqlite's own recorded source URLs for each image (slow, network-bound -- can take a very long time re-fetching a large fraction of a corpus; MD5-verified against the identity already on record before trusting any download, so a site serving different content at an old URL now can't silently reintroduce a wrong image under someone else's identity). Resumable at every stage. --output-dir's crawl.sqlite is read-only unless --finalize is passed.");
recoverCorruptedImagesCommand.Options.Add(outputDirOption);
recoverCorruptedImagesCommand.Options.Add(stagingDirOption);
recoverCorruptedImagesCommand.Options.Add(danbooruLoginOption);
recoverCorruptedImagesCommand.Options.Add(danbooruApiKeyOption);
recoverCorruptedImagesCommand.Options.Add(gelbooruApiKeyOption);
recoverCorruptedImagesCommand.Options.Add(gelbooruUserIdOption);
recoverCorruptedImagesCommand.Options.Add(rateDanbooruOption);
recoverCorruptedImagesCommand.Options.Add(rateGelbooruOption);
recoverCorruptedImagesCommand.Options.Add(recoveryConcurrencyOption);
recoverCorruptedImagesCommand.Options.Add(finalizeOption);
recoverCorruptedImagesCommand.SetAction(async (parseResult, cancellationToken) =>
{
    var outputDirectory = parseResult.GetRequiredValue(outputDirOption);
    var stagingDirectory = parseResult.GetRequiredValue(stagingDirOption);
    var rateDanbooru = parseResult.GetRequiredValue(rateDanbooruOption);
    var rateGelbooru = parseResult.GetRequiredValue(rateGelbooruOption);
    var concurrency = parseResult.GetRequiredValue(recoveryConcurrencyOption);
    var finalize = parseResult.GetRequiredValue(finalizeOption);

    if (string.Equals(Path.GetFullPath(stagingDirectory), Path.GetFullPath(outputDirectory), StringComparison.Ordinal))
    {
        Console.Error.WriteLine("--staging-dir must not be the same directory as --output-dir.");
        return 1;
    }

    var brokenPixelPath = Path.Combine(outputDirectory, "images.bin");
    int inputSize;
    using (var headerStream = File.OpenRead(brokenPixelPath))
    using (var headerReader = new BinaryReader(headerStream))
    {
        headerReader.ReadInt32(); // magic
        headerReader.ReadInt32(); // row count -- not trusted here, DetectValidPrefixLength below is the source of truth
        inputSize = headerReader.ReadInt32();
    }

    using var db = await CrawlDatabase.OpenOrCreateAsync(outputDirectory, cancellationToken);

    Console.WriteLine("Computing recovery plan from crawl.sqlite...");
    var plan = await DatasetRecovery.ComputePlanAsync(db, cancellationToken);
    Console.WriteLine($"Target: {plan.OrderedMd5s.Count} images ({plan.SoleSourceMd5s.Count} sole-unsupported-media-source excluded, {plan.ExtraSourcesToStrip.Count} extra bad source row(s) to strip on finalize).");

    // --output-dir is never written to by this command (the whole point is recovering
    // INTO --staging-dir without risking further damage there), so the valid-prefix
    // length can never change between runs of the same recovery -- cached in
    // --staging-dir so a restart (this command is meant to be interrupted and re-run
    // freely) doesn't re-pay a full read of the damaged cache's multi-terabyte valid
    // portion just to re-derive an answer that can only ever come out the same.
    Directory.CreateDirectory(stagingDirectory);
    var validPrefixCachePath = Path.Combine(stagingDirectory, "valid-prefix-length.txt");
    int validPrefixLength;
    if (File.Exists(validPrefixCachePath) && int.TryParse(File.ReadAllText(validPrefixCachePath), out var cachedValidPrefixLength))
    {
        validPrefixLength = cachedValidPrefixLength;
        Console.WriteLine($"Using cached valid-prefix length from a prior run: {validPrefixLength} row(s).");
    }
    else
    {
        Console.WriteLine("Checking how much of the damaged cache still has real pixel content...");
        validPrefixLength = PreprocessedDatasetCacheCompactor.DetectValidPrefixLength(outputDirectory);
        File.WriteAllText(validPrefixCachePath, validPrefixLength.ToString());
    }
    Console.WriteLine($"{validPrefixLength} of {plan.OrderedMd5s.Count} row(s) still valid; {plan.OrderedMd5s.Count - validPrefixLength} need re-downloading.");

    if (validPrefixLength > 0)
    {
        Console.WriteLine("Copying the still-valid prefix into the staging cache (source is never modified)...");
        AnsiConsole.Progress().Columns(ProgressBarColumns.Default).Start(ctx =>
        {
            var task = ctx.AddTask("Copying valid rows...");
            task.MaxValue = validPrefixLength;
            PreprocessedDatasetCacheCompactor.CopyPrefix(
                outputDirectory, stagingDirectory, validPrefixLength,
                onProgress: (done, total) => { task.MaxValue = total; task.Value = done; },
                cancellationToken);
        });
    }

    Console.WriteLine("Re-downloading the rest from their recorded source URLs -- this can take a very long time. Safe to interrupt (Ctrl-C) and re-run; it resumes automatically.");
    var clients = BuildClients(
        ["danbooru", "gelbooru"], httpClient, rateDanbooru, rateGelbooru,
        parseResult.GetValue(danbooruLoginOption), parseResult.GetValue(danbooruApiKeyOption),
        parseResult.GetValue(gelbooruApiKeyOption), parseResult.GetValue(gelbooruUserIdOption));
    var rateLimiterBySite = clients.ToDictionary(kv => kv.Key, kv => kv.Value.RateLimiter, StringComparer.Ordinal);
    var ratesBySite = new Dictionary<string, double> { ["danbooru"] = rateDanbooru, ["gelbooru"] = rateGelbooru }
        .Where(kv => clients.ContainsKey(kv.Key))
        .ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal);
    var sourceUrls = await db.GetAllImageSourceUrlsAsync(cancellationToken);

    var fillResult = await AnsiConsole.Progress().Columns(ProgressBarColumns.Default).StartAsync(async ctx =>
    {
        var task = ctx.AddTask("Re-downloading missing rows...");
        task.MaxValue = plan.OrderedMd5s.Count;
        return await DatasetRecovery.FillRemainingAsync(
            stagingDirectory, outputDirectory, inputSize, plan.OrderedMd5s, sourceUrls, httpClient, rateLimiterBySite, ratesBySite, concurrency,
            onProgress: (done, total) => { task.MaxValue = total; task.Value = done; },
            cancellationToken);
    });

    Console.WriteLine($"Recovered {fillResult.WrittenMd5s.Count} of {plan.OrderedMd5s.Count} images ({fillResult.PermanentlyFailedMd5s.Count} permanently unrecoverable -- see '{Path.Combine(stagingDirectory, DatasetRecovery.FailuresLogFileName)}').");

    Console.WriteLine("Verifying the staging cache (row count + full structural scan)...");
    var stagingValidCount = PreprocessedDatasetCacheCompactor.DetectValidPrefixLength(stagingDirectory);
    if (stagingValidCount != fillResult.WrittenMd5s.Count)
    {
        Console.Error.WriteLine($"Staging cache at '{stagingDirectory}' has {stagingValidCount} row(s) with real content but expected {fillResult.WrittenMd5s.Count} -- NOT finalizing. Something is wrong; investigate before re-running.");
        return 1;
    }
    Console.WriteLine("Staging cache verified: every row has real content and the count matches.");

    if (!finalize)
    {
        Console.WriteLine($"Staging cache is ready at '{stagingDirectory}'. Re-run this same command with --finalize to swap it into '{outputDirectory}' and reconcile crawl.sqlite.");
        return 0;
    }

    Console.WriteLine("Finalizing: swapping the staging cache into --output-dir...");
    var brokenTagRowPath = Path.Combine(outputDirectory, "tag_rows.sqlite");
    var backupPixelPath = brokenPixelPath + ".broken-pre-recovery.bak";
    var backupTagRowPath = brokenTagRowPath + ".broken-pre-recovery.bak";
    // No overwrite: a pre-existing backup means an earlier --finalize attempt already
    // got this far and something went wrong after -- investigate rather than silently
    // clobber whatever that attempt preserved.
    File.Move(brokenPixelPath, backupPixelPath);
    File.Move(brokenTagRowPath, backupTagRowPath);
    File.Move(Path.Combine(stagingDirectory, "images.bin"), brokenPixelPath);
    File.Move(Path.Combine(stagingDirectory, "tag_rows.sqlite"), brokenTagRowPath);
    Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools(); // see PreprocessedDatasetCacheCompactor.SwapIn's own doc comment on why this matters after a rename-based swap
    var resumeIndexPath = Path.Combine(outputDirectory, "images.bin.resume");
    var stagingResumeIndexPath = Path.Combine(stagingDirectory, "images.bin.resume");
    if (File.Exists(stagingResumeIndexPath))
        File.Move(stagingResumeIndexPath, resumeIndexPath, overwrite: true);
    else if (File.Exists(resumeIndexPath))
        File.Delete(resumeIndexPath);

    Console.WriteLine("Reconciling crawl.sqlite...");
    var md5sToDelete = plan.SoleSourceMd5s.Concat(fillResult.PermanentlyFailedMd5s).ToList();
    var newCacheRowIndexByMd5 = fillResult.WrittenMd5s
        .Select((md5, index) => (md5, index))
        .ToDictionary(t => t.md5, t => t.index, StringComparer.Ordinal);
    var extraSourcesToStrip = plan.ExtraSourcesToStrip.Select(s => (s.Site, s.PostId)).ToList();
    await db.ApplyRecoveryReconciliationAsync(md5sToDelete, extraSourcesToStrip, newCacheRowIndexByMd5, cancellationToken);

    Console.WriteLine($"Done. '{backupPixelPath}' and '{backupTagRowPath}' hold the damaged pre-recovery files -- kept, not deleted; remove them yourself once you've confirmed the recovery looks right. Run rebuild-vocabulary against this --output-dir to reconcile tag_vocabulary.json's per-tag counts.");
    return 0;
});

var rootCommand = new RootCommand("unbooru-tagger booru crawler (downloads Danbooru/Gelbooru images+tags directly into a trainable dataset directory)")
{
    surveyCommand,
    crawlCommand,
    refreshCommand,
    shrinkCacheCommand,
    pruneAliasedTagsCommand,
    detectDeprecatedRedirectsCommand,
    rebuildVocabularyCommand,
    pruneUnsupportedMediaCommand,
    recoverCorruptedImagesCommand
};

return await rootCommand.Parse(args).InvokeAsync();
