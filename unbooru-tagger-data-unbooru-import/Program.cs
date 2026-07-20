using System.CommandLine;
using Spectre.Console;
using UnbooruTagger.Data;

var connectionStringOption = new Option<string>("--connection-string") { Description = "unbooru SQL Server connection string", Required = true };
var outOption = new Option<string>("--out") { Description = "Output directory", Required = true };

var tagsOption = new Option<string[]>("--tags")
{
    Description = "Target tag name(s) — images with ANY of these are positives",
    Required = true,
    AllowMultipleArgumentsPerToken = true
};
var smallMaxImagesOption = new Option<int?>("--max-images") { Description = "Cap on positive images pulled (all matches if omitted)" };

var buildSmallCommand = new Command("build-small-dataset", "Pull images with target tags + an equal number of random images without them");
buildSmallCommand.Options.Add(tagsOption);
buildSmallCommand.Options.Add(smallMaxImagesOption);
buildSmallCommand.Options.Add(connectionStringOption);
buildSmallCommand.Options.Add(outOption);
buildSmallCommand.SetAction(async (parseResult, cancellationToken) =>
{
    var outputDirectory = parseResult.GetRequiredValue(outOption);
    using var context = UnbooruContextFactory.Create(parseResult.GetRequiredValue(connectionStringOption));

    var manifest = await AnsiConsole.Progress()
        .Columns(ProgressBarColumns.Default)
        .StartAsync(ctx => SmallDatasetBuilder.BuildAsync(
            context,
            parseResult.GetRequiredValue(tagsOption),
            outputDirectory,
            parseResult.GetValue(smallMaxImagesOption),
            progress: ProgressBarColumns.AddTask(ctx, "Writing images"),
            cancellationToken: cancellationToken));

    Console.WriteLine($"Wrote {manifest.Entries.Count} images to '{outputDirectory}'.");
    return 0;
});

var inputSizeOption = new Option<int>("--input-size") { Description = "Square input resolution to preprocess images to", DefaultValueFactory = _ => 224 };
var largeMaxImagesOption = new Option<int?>("--max-images") { Description = "Cap on total images pulled (all images if omitted)" };
var minImagesPerTagOption = new Option<int>("--min-images-per-tag")
{
    Description = "When --max-images caps the corpus, images to reserve per tag (rarest first) so every known tag gets a fair shot at training",
    DefaultValueFactory = _ => 15
};
var minTagImagesOption = new Option<int>("--min-tag-images")
{
    Description = "Only include tags that appear on at least this many images across the full corpus; images keep any other tags they have",
    DefaultValueFactory = _ => 100
};
var pageSizeOption = new Option<int>("--page-size") { Description = "Images pulled from the DB per round-trip; a run resumes after this many images on a crash/connection drop", DefaultValueFactory = _ => 500 };
var vocabCompactionIntervalOption = new Option<int>("--vocab-compact-interval")
{
    Description = "Pages between full tag_vocabulary.json compactions (the delta log is still checkpointed every page regardless)",
    DefaultValueFactory = _ => 20
};

var buildLargeCommand = new Command("build-large-cache", "Preprocess the full (or a capped) corpus into a fast-loading training cache. Re-running against the same --out directory resumes an interrupted run.");
buildLargeCommand.Options.Add(connectionStringOption);
buildLargeCommand.Options.Add(outOption);
buildLargeCommand.Options.Add(inputSizeOption);
buildLargeCommand.Options.Add(largeMaxImagesOption);
buildLargeCommand.Options.Add(minImagesPerTagOption);
buildLargeCommand.Options.Add(minTagImagesOption);
buildLargeCommand.Options.Add(pageSizeOption);
buildLargeCommand.Options.Add(vocabCompactionIntervalOption);
buildLargeCommand.SetAction(async (parseResult, cancellationToken) =>
{
    var outputDirectory = parseResult.GetRequiredValue(outOption);
    var connectionString = parseResult.GetRequiredValue(connectionStringOption);
    using var context = UnbooruContextFactory.Create(connectionString);

    await AnsiConsole.Progress()
        .Columns(ProgressBarColumns.LargeCacheColumns)
        .StartAsync(ctx => LargeDatasetPreprocessor.BuildAsync(
            context,
            outputDirectory,
            parseResult.GetRequiredValue(inputSizeOption),
            parseResult.GetValue(largeMaxImagesOption),
            parseResult.GetRequiredValue(minImagesPerTagOption),
            parseResult.GetRequiredValue(minTagImagesOption),
            ProgressBarColumns.AddLargeCacheTasks(ctx),
            cancellationToken,
            parseResult.GetRequiredValue(pageSizeOption),
            parseResult.GetRequiredValue(vocabCompactionIntervalOption),
            // A separate connection per blob-fetch chunk (see BuildAsync) so the slow
            // part of a page's fetch -- transferring full-resolution image bytes --
            // isn't serialized behind a single connection.
            contextFactory: () => UnbooruContextFactory.Create(connectionString)));

    Console.WriteLine($"Done. Cache written to '{outputDirectory}'.");
    return 0;
});

var rootCommand = new RootCommand("unbooru-tagger data pipeline (pulls training data from unbooru's database)")
{
    buildSmallCommand,
    buildLargeCommand
};

return await rootCommand.Parse(args).InvokeAsync();
