using System.CommandLine;
using Spectre.Console;
using UnbooruTagger.Core.Dataset;
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
        Console.Error.WriteLine($"Failed to fetch Danbooru's active tag aliases: {ex.Message}");
        Console.Error.WriteLine($"'{commandName}' depends on this to correctly merge/reconcile cross-site tag aliases (e.g. head_pat/headpat) — exiting without making any changes rather than risk a silently wrong result. Try again once connectivity is restored.");
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

var danbooruLoginOption = new Option<string?>("--danbooru-login") { Description = "Danbooru username (paired with --danbooru-api-key) for a higher rate-limit tier" };
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

    Console.WriteLine($"Surveyed {summary.TotalTagsSeen} tags; {summary.EligibleTagCount} eligible at --min-images {minImages} ({summary.ExcludedTagCount} excluded via '{TagExclusions.FileName}').");
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

var rootCommand = new RootCommand("unbooru-tagger booru crawler (downloads Danbooru/Gelbooru images+tags directly into a trainable dataset directory)")
{
    surveyCommand,
    crawlCommand,
    refreshCommand,
    shrinkCacheCommand
};

return await rootCommand.Parse(args).InvokeAsync();
