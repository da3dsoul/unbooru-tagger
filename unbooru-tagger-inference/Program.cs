using System.CommandLine;
using UnbooruTagger.Inference.Commands;

var modelDirOption = new Option<string>("--model-dir")
{
    Description = "Directory containing image_encoder.onnx, tag_vocabulary.json and tag_embeddings.bin",
    DefaultValueFactory = _ => "./model"
};

var thresholdOption = new Option<float>("--threshold")
{
    Description = "Minimum confidence required to report a tag",
    DefaultValueFactory = _ => 0.5f
};

var imageArgument = new Argument<string>("image") { Description = "Path to the image to tag" };
var directoryArgument = new Argument<string>("directory") { Description = "Directory of images to tag" };
var tagArgument = new Argument<string>("tag") { Description = "Tag to localize" };
var outputOption = new Option<string>("--out")
{
    Description = "Where to write the heatmap overlay PNG",
    DefaultValueFactory = _ => "heatmap.png"
};

var boxThresholdOption = new Option<float>("--box-threshold")
{
    Description = "Absolute floor: minimum per-location confidence required to include a spatial cell in a bounding box",
    DefaultValueFactory = _ => 0.5f
};

var boxPercentileOption = new Option<float>("--box-percentile")
{
    Description = "Additionally cut within each tag's own heatmap range (0 = its weakest cell, 1 = only its single strongest) to tighten boxes around each tag's peak",
    DefaultValueFactory = _ => 0.6f
};

var detectOutputOption = new Option<string?>("--out")
{
    Description = "If set, also render the detected boxes onto the image and write it as a PNG here"
};

var tagCommand = new Command("tag", "Score a single image against the tag vocabulary");
tagCommand.Arguments.Add(imageArgument);
tagCommand.Options.Add(modelDirOption);
tagCommand.Options.Add(thresholdOption);
tagCommand.SetAction(parseResult => TagCommandHandler.Run(
    parseResult.GetRequiredValue(modelDirOption),
    parseResult.GetRequiredValue(imageArgument),
    parseResult.GetRequiredValue(thresholdOption)));

var tagBatchCommand = new Command("tag-batch", "Score every image in a directory against the tag vocabulary");
tagBatchCommand.Arguments.Add(directoryArgument);
tagBatchCommand.Options.Add(modelDirOption);
tagBatchCommand.Options.Add(thresholdOption);
tagBatchCommand.SetAction(parseResult => TagBatchCommandHandler.Run(
    parseResult.GetRequiredValue(modelDirOption),
    parseResult.GetRequiredValue(directoryArgument),
    parseResult.GetRequiredValue(thresholdOption)));

var heatmapCommand = new Command("heatmap", "Render a rough localization heatmap for one tag on one image");
heatmapCommand.Arguments.Add(imageArgument);
heatmapCommand.Arguments.Add(tagArgument);
heatmapCommand.Options.Add(modelDirOption);
heatmapCommand.Options.Add(outputOption);
heatmapCommand.SetAction(parseResult => HeatmapCommandHandler.Run(
    parseResult.GetRequiredValue(modelDirOption),
    parseResult.GetRequiredValue(imageArgument),
    parseResult.GetRequiredValue(tagArgument),
    parseResult.GetRequiredValue(outputOption)));

var detectCommand = new Command("detect", "Tag an image and report a rough bounding box per detected tag");
detectCommand.Arguments.Add(imageArgument);
detectCommand.Options.Add(modelDirOption);
detectCommand.Options.Add(thresholdOption);
detectCommand.Options.Add(boxThresholdOption);
detectCommand.Options.Add(boxPercentileOption);
detectCommand.Options.Add(detectOutputOption);
detectCommand.SetAction(parseResult => DetectCommandHandler.Run(
    parseResult.GetRequiredValue(modelDirOption),
    parseResult.GetRequiredValue(imageArgument),
    parseResult.GetRequiredValue(thresholdOption),
    parseResult.GetRequiredValue(boxThresholdOption),
    parseResult.GetRequiredValue(boxPercentileOption),
    parseResult.GetValue(detectOutputOption)));

var rootCommand = new RootCommand("unbooru-tagger inference CLI")
{
    tagCommand,
    tagBatchCommand,
    heatmapCommand,
    detectCommand
};

return rootCommand.Parse(args).Invoke();
