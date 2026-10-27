using UnbooruTagger.Core.Encoding;
using System.CommandLine;
using UnbooruTagger.Inference.Commands;

var modelDirOption = new Option<string>("--model-dir")
{
    Description = "Directory containing image_encoder.onnx, tag_vocabulary.json and tag_embeddings.bin",
    DefaultValueFactory = _ => "./model"
};

var thresholdOption = new Option<float?>("--threshold")
{
    Description = "Minimum confidence required to report a tag. Omit to use each tag's calibrated threshold from the model bundle's tag_thresholds.json (written by the training CLI's evaluate command), falling back to 0.5 for uncalibrated tags or bundles without that file."
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

var minHeatmapRangeOption = new Option<float>("--min-heatmap-range")
{
    Description = "Suppress all boxes for a tag whose heatmap's own max-min spread sits below this — a near-flat heatmap has no real spatial signal to cut into, so a box built from it is noise, not localization",
    DefaultValueFactory = _ => 0.05f
};

var minBoxFractionOption = new Option<float>("--min-box-fraction")
{
    Description = "Drop any box whose connected region covers less than this fraction of the heatmap grid -- isolated specks are noise, not localization",
    DefaultValueFactory = _ => 0.002f
};

var maxBoxesPerTagOption = new Option<int>("--max-boxes-per-tag")
{
    Description = "Keep at most this many boxes per tag, strongest peak first. 0 keeps every box",
    DefaultValueFactory = _ => 3
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
    parseResult.GetValue(thresholdOption)));

var batchSizeOption = new Option<int>("--batch-size")
{
    Description = "Images per model run. Needs a bundle exported with a batch dimension; older ones run one at a time",
    DefaultValueFactory = _ => 16
};

var workersOption = new Option<int>("--workers")
{
    Description = "Threads for decoding images and for per-image scoring/box extraction (each). 0 = half the logical CPUs",
    DefaultValueFactory = _ => 0
};

var deviceOption = new Option<InferenceDevice>("--device")
{
    Description = "Where the image model runs: Auto (CUDA GPU 0 if ONNX Runtime can load it, else CPU), Cuda, or Cpu",
    DefaultValueFactory = _ => InferenceDevice.Auto
};

var tagBatchCommand = new Command("tag-batch", "Score every image in a directory against the tag vocabulary");
tagBatchCommand.Arguments.Add(directoryArgument);
tagBatchCommand.Options.Add(modelDirOption);
tagBatchCommand.Options.Add(thresholdOption);
tagBatchCommand.Options.Add(batchSizeOption);
tagBatchCommand.Options.Add(workersOption);
tagBatchCommand.Options.Add(deviceOption);
tagBatchCommand.SetAction(parseResult => TagBatchCommandHandler.Run(
    parseResult.GetRequiredValue(modelDirOption),
    parseResult.GetRequiredValue(directoryArgument),
    parseResult.GetValue(thresholdOption),
    parseResult.GetRequiredValue(batchSizeOption),
    parseResult.GetRequiredValue(workersOption),
    parseResult.GetRequiredValue(deviceOption)));

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
detectCommand.Options.Add(minHeatmapRangeOption);
detectCommand.Options.Add(minBoxFractionOption);
detectCommand.Options.Add(maxBoxesPerTagOption);
detectCommand.Options.Add(detectOutputOption);
detectCommand.SetAction(parseResult => DetectCommandHandler.Run(
    parseResult.GetRequiredValue(modelDirOption),
    parseResult.GetRequiredValue(imageArgument),
    parseResult.GetValue(thresholdOption),
    parseResult.GetRequiredValue(boxThresholdOption),
    parseResult.GetRequiredValue(boxPercentileOption),
    parseResult.GetRequiredValue(minHeatmapRangeOption),
    parseResult.GetRequiredValue(minBoxFractionOption),
    parseResult.GetRequiredValue(maxBoxesPerTagOption),
    parseResult.GetValue(detectOutputOption)));

var goldArgument = new Argument<string>("gold") { Description = "Gold set file (gold.json); its images are resolved relative to it" };
var renderOutDirOption = new Option<string>("--out-dir") { Description = "Directory to write one PNG per gold image", Required = true };
var optionalModelDirOption = new Option<string?>("--model-dir")
{
    Description = "If set, also run detect with this model on each image and render it beside the gold boxes"
};

var allTagsOption = new Option<bool>("--all-tags") { Description = "Show every tag the model detects, not just the ones the gold set boxes on that image" };
var pngOption = new Option<bool>("--png") { Description = "With a model, also write the side-by-side PNGs with the boxes drawn in (they're otherwise only written to boxes.json, for gold/compare.html's interactive view)" };
var renderGoldCommand = new Command("render-gold", "Render a gold set's hand-checked boxes the way detect renders, optionally side by side with a model's detect output, for manual comparison");
renderGoldCommand.Arguments.Add(goldArgument);
renderGoldCommand.Options.Add(renderOutDirOption);
renderGoldCommand.Options.Add(optionalModelDirOption);
renderGoldCommand.Options.Add(thresholdOption);
renderGoldCommand.Options.Add(boxThresholdOption);
renderGoldCommand.Options.Add(boxPercentileOption);
renderGoldCommand.Options.Add(minHeatmapRangeOption);
renderGoldCommand.Options.Add(minBoxFractionOption);
renderGoldCommand.Options.Add(maxBoxesPerTagOption);
renderGoldCommand.Options.Add(allTagsOption);
renderGoldCommand.Options.Add(pngOption);
renderGoldCommand.SetAction(parseResult => RenderGoldCommandHandler.Run(
    parseResult.GetRequiredValue(goldArgument),
    parseResult.GetRequiredValue(renderOutDirOption),
    parseResult.GetValue(optionalModelDirOption),
    parseResult.GetValue(thresholdOption),
    parseResult.GetRequiredValue(boxThresholdOption),
    parseResult.GetRequiredValue(boxPercentileOption),
    parseResult.GetRequiredValue(minHeatmapRangeOption),
    parseResult.GetRequiredValue(minBoxFractionOption),
    parseResult.GetRequiredValue(maxBoxesPerTagOption),
    parseResult.GetValue(allTagsOption),
    parseResult.GetValue(pngOption)));

var resultsOption = new Option<string>("--out")
{
    Description = "Where to write the results, one JSON object per image per line (JSONL)",
    Required = true
};
var renderDirOption = new Option<string?>("--render-dir") { Description = "If set, also write each image with its boxes drawn on it here (PNG)" };
var noBoxesOption = new Option<bool>("--no-boxes") { Description = "Whole-image tags only: skip heatmaps and box extraction" };
var recursiveOption = new Option<bool>("--recursive") { Description = "Include images in subdirectories" };

var detectBatchCommand = new Command("detect-batch", "Tag and detect every image in a directory in one pass, batched on the GPU when available");
detectBatchCommand.Arguments.Add(directoryArgument);
detectBatchCommand.Options.Add(modelDirOption);
detectBatchCommand.Options.Add(resultsOption);
detectBatchCommand.Options.Add(renderDirOption);
detectBatchCommand.Options.Add(noBoxesOption);
detectBatchCommand.Options.Add(recursiveOption);
detectBatchCommand.Options.Add(batchSizeOption);
detectBatchCommand.Options.Add(workersOption);
detectBatchCommand.Options.Add(deviceOption);
detectBatchCommand.Options.Add(thresholdOption);
detectBatchCommand.Options.Add(boxThresholdOption);
detectBatchCommand.Options.Add(boxPercentileOption);
detectBatchCommand.Options.Add(minHeatmapRangeOption);
detectBatchCommand.Options.Add(minBoxFractionOption);
detectBatchCommand.Options.Add(maxBoxesPerTagOption);
detectBatchCommand.SetAction(parseResult => DetectBatchCommandHandler.Run(
    parseResult.GetRequiredValue(modelDirOption),
    parseResult.GetRequiredValue(directoryArgument),
    parseResult.GetRequiredValue(resultsOption),
    parseResult.GetValue(renderDirOption),
    !parseResult.GetValue(noBoxesOption),
    parseResult.GetValue(recursiveOption),
    parseResult.GetRequiredValue(batchSizeOption),
    parseResult.GetRequiredValue(workersOption),
    parseResult.GetRequiredValue(deviceOption),
    parseResult.GetValue(thresholdOption),
    parseResult.GetRequiredValue(boxThresholdOption),
    parseResult.GetRequiredValue(boxPercentileOption),
    parseResult.GetRequiredValue(minHeatmapRangeOption),
    parseResult.GetRequiredValue(minBoxFractionOption),
    parseResult.GetRequiredValue(maxBoxesPerTagOption)));

var rootCommand = new RootCommand("unbooru-tagger inference CLI")
{
    tagCommand,
    tagBatchCommand,
    detectBatchCommand,
    heatmapCommand,
    detectCommand,
    renderGoldCommand
};

return rootCommand.Parse(args).Invoke();
