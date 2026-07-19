using System.CommandLine;
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

    var manifest = await SmallDatasetBuilder.BuildAsync(
        context,
        parseResult.GetRequiredValue(tagsOption),
        outputDirectory,
        parseResult.GetValue(smallMaxImagesOption),
        cancellationToken: cancellationToken);

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

var buildLargeCommand = new Command("build-large-cache", "Preprocess the full (or a capped) corpus into a fast-loading training cache");
buildLargeCommand.Options.Add(connectionStringOption);
buildLargeCommand.Options.Add(outOption);
buildLargeCommand.Options.Add(inputSizeOption);
buildLargeCommand.Options.Add(largeMaxImagesOption);
buildLargeCommand.Options.Add(minImagesPerTagOption);
buildLargeCommand.SetAction(async (parseResult, cancellationToken) =>
{
    var outputDirectory = parseResult.GetRequiredValue(outOption);
    using var context = UnbooruContextFactory.Create(parseResult.GetRequiredValue(connectionStringOption));

    var progress = new Progress<int>(count =>
    {
        if (count % 100 == 0)
            Console.WriteLine($"Preprocessed {count} images...");
    });

    await LargeDatasetPreprocessor.BuildAsync(
        context,
        outputDirectory,
        parseResult.GetRequiredValue(inputSizeOption),
        parseResult.GetValue(largeMaxImagesOption),
        parseResult.GetRequiredValue(minImagesPerTagOption),
        progress,
        cancellationToken);

    Console.WriteLine($"Done. Cache written to '{outputDirectory}'.");
    return 0;
});

var rootCommand = new RootCommand("unbooru-tagger data pipeline (pulls training data from unbooru's database)")
{
    buildSmallCommand,
    buildLargeCommand
};

return await rootCommand.Parse(args).InvokeAsync();
