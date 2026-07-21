using System.CommandLine;
using Spectre.Console;
using UnbooruTagger.Crawler;

var outputDirOption = new Option<string>("--output-dir") { Description = "Dataset directory — gets images.bin/tag_rows.jsonl/tag_vocabulary.json (same format build-large-cache produces) plus crawl.sqlite", Required = true };
var minImagesOption = new Option<int>("--min-images") { Description = "Only crawl tags with at least this many posts on at least one site", DefaultValueFactory = _ => 500 };
var maxImagesOption = new Option<int>("--max-images") { Description = "Cap on images pulled per eligible tag (combined across both sites)", DefaultValueFactory = _ => 1000 };

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
            return await TagSurveyor.SurveyAsync(db, clients.Values.ToList(), minImages, maxImages, progress, cancellationToken);
        });

    foreach (var note in stopNotes)
        AnsiConsole.MarkupLine($"[grey]{Markup.Escape(note)}[/]");
    if (stopNotes.Any(n => n.Contains("stopped after 0 eligible")))
        AnsiConsole.MarkupLine("[yellow]A site found zero eligible tags — if that's unexpected, the site's tag-list API may not actually be honoring the sort-by-count-descending request, so this stopped on the very first (effectively random) tag instead of the least popular eligible one.[/]");

    Console.WriteLine($"Surveyed {summary.TotalTagsSeen} tags; {summary.EligibleTagCount} eligible at --min-images {minImages}.");
    PrintEstimate(new CrawlEstimate(summary.EligibleTagCount, summary.EstimatedImageSlots, 0, TimeSpan.Zero));
    return 0;
});

var negativeTargetOption = new Option<int?>("--negative-target") { Description = "Non-tagged images each eligible tag should end up with (default 2x --min-images — see plan notes on dedup bias)" };
var inputSizeOption = new Option<int>("--input-size") { Description = "Square input resolution to preprocess images to", DefaultValueFactory = _ => 224 };
var vocabCompactIntervalOption = new Option<int>("--vocab-compact-interval") { Description = "Pages between full tag_vocabulary.json compactions (the delta log is still checkpointed every page regardless, same as build-large-cache's option of the same name)", DefaultValueFactory = _ => 20 };

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
crawlCommand.SetAction(async (parseResult, cancellationToken) =>
{
    var outputDirectory = parseResult.GetRequiredValue(outputDirOption);
    var minImages = parseResult.GetRequiredValue(minImagesOption);
    var maxImages = parseResult.GetRequiredValue(maxImagesOption);
    var sites = parseResult.GetRequiredValue(sitesOption);
    var inputSize = parseResult.GetRequiredValue(inputSizeOption);
    var rateDanbooru = parseResult.GetRequiredValue(rateDanbooruOption);
    var rateGelbooru = parseResult.GetRequiredValue(rateGelbooruOption);
    var negativeTarget = parseResult.GetValue(negativeTargetOption) ?? minImages * 2;
    var vocabCompactInterval = parseResult.GetRequiredValue(vocabCompactIntervalOption);

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

    var pageSizeBySite = clients.ToDictionary(kv => kv.Key, kv => kv.Value.PageSize);
    var rateBySite = new Dictionary<string, double> { ["danbooru"] = rateDanbooru, ["gelbooru"] = rateGelbooru }
        .Where(kv => clients.ContainsKey(kv.Key))
        .ToDictionary(kv => kv.Key, kv => kv.Value);

    var estimate = DatasetCrawler.Estimate(allTags, minImages, maxImages, pageSizeBySite, rateBySite, tagSurveyRequestsMade: 0);
    Console.WriteLine("Estimate (recomputed from the last survey-tags run):");
    PrintEstimate(estimate);

    var shortfalls = await AnsiConsole.Progress()
        .Columns(ProgressBarColumns.Default)
        .StartAsync(async ctx =>
        {
            var progress = ProgressBarColumns.AddCrawlTasks(ctx);
            return await DatasetCrawler.RunAsync(
                db, clients, httpClient, outputDirectory, inputSize,
                minImages, maxImages, negativeTarget, vocabCompactInterval,
                progress, cancellationToken);
        });

    if (shortfalls.Count > 0)
    {
        AnsiConsole.MarkupLine($"[yellow]{shortfalls.Count} eligible tag(s) fell short of --max-images {maxImages} — both sites ran out of posts for these before dedup (md5 + perceptual hash) let the tag reach quota:[/]");
        var shortfallTable = new Table().Border(TableBorder.Rounded);
        shortfallTable.AddColumn("Tag");
        shortfallTable.AddColumn("Achieved");
        shortfallTable.AddColumn("Target");
        foreach (var shortfall in shortfalls)
            shortfallTable.AddRow(shortfall.TagName, shortfall.Achieved.ToString(), shortfall.Target.ToString());
        AnsiConsole.Write(shortfallTable);
    }

    Console.WriteLine($"Done. Dataset written to '{outputDirectory}'.");
    return 0;
});

var rootCommand = new RootCommand("unbooru-tagger booru crawler (downloads Danbooru/Gelbooru images+tags directly into a trainable dataset directory)")
{
    surveyCommand,
    crawlCommand
};

return await rootCommand.Parse(args).InvokeAsync();
