using System.CommandLine;
using UnbooruTagger.Training.Commands;

var checkpointDirOption = new Option<string>("--checkpoint-dir")
{
    Description = "Directory holding the TorchSharp training checkpoint (image_tower.dat, model_config.json, tag_vocabulary.json, tag_embeddings.bin)",
    DefaultValueFactory = _ => "./checkpoint"
};

var manifestOption = new Option<string>("--manifest") { Description = "Path to the dataset manifest JSON", Required = true };
var embeddingDimOption = new Option<int>("--embedding-dim") { Description = "Tag/image embedding dimension", DefaultValueFactory = _ => 512 };
var inputSizeOption = new Option<int>("--input-size") { Description = "Square input resolution fed to the image tower", DefaultValueFactory = _ => 224 };
var epochsOption = new Option<int>("--epochs") { Description = "Number of passes over the dataset", DefaultValueFactory = _ => 5 };
var batchSizeOption = new Option<int>("--batch-size") { DefaultValueFactory = _ => 16 };
var learningRateOption = new Option<double>("--lr") { DefaultValueFactory = _ => 1e-4 };

var trainCommand = new Command("train", "Full/periodic fine-tune pass over a dataset manifest");
trainCommand.Options.Add(manifestOption);
trainCommand.Options.Add(checkpointDirOption);
trainCommand.Options.Add(embeddingDimOption);
trainCommand.Options.Add(inputSizeOption);
trainCommand.Options.Add(epochsOption);
trainCommand.Options.Add(batchSizeOption);
trainCommand.Options.Add(learningRateOption);
trainCommand.SetAction(parseResult => TrainCommandHandler.Run(
    parseResult.GetRequiredValue(manifestOption),
    parseResult.GetRequiredValue(checkpointDirOption),
    parseResult.GetRequiredValue(embeddingDimOption),
    parseResult.GetRequiredValue(inputSizeOption),
    parseResult.GetRequiredValue(epochsOption),
    parseResult.GetRequiredValue(batchSizeOption),
    parseResult.GetRequiredValue(learningRateOption)));

var tagArgument = new Argument<string>("tag") { Description = "The new tag to add" };
var imagesOption = new Option<string>("--images") { Description = "Dataset manifest of the newly tagged images", Required = true };
var stepsOption = new Option<int>("--steps") { Description = "Gradient steps to fine-tune the new row for", DefaultValueFactory = _ => 200 };
var addTagLearningRateOption = new Option<double>("--lr") { DefaultValueFactory = _ => 1e-3 };
var minImageThresholdOption = new Option<int>("--min-image-threshold")
{
    Description = "Images needed before the tag is promoted from warm-start-only to trained",
    DefaultValueFactory = _ => 15
};

var addTagCommand = new Command("add-tag", "Warm-start and fine-tune a single new tag row, image encoder frozen");
addTagCommand.Arguments.Add(tagArgument);
addTagCommand.Options.Add(checkpointDirOption);
addTagCommand.Options.Add(imagesOption);
addTagCommand.Options.Add(stepsOption);
addTagCommand.Options.Add(addTagLearningRateOption);
addTagCommand.Options.Add(minImageThresholdOption);
addTagCommand.SetAction(parseResult => AddTagCommandHandler.Run(
    parseResult.GetRequiredValue(checkpointDirOption),
    parseResult.GetRequiredValue(tagArgument),
    parseResult.GetRequiredValue(imagesOption),
    parseResult.GetRequiredValue(stepsOption),
    parseResult.GetRequiredValue(addTagLearningRateOption),
    parseResult.GetRequiredValue(minImageThresholdOption),
    warmStartEmbedder: null));

var modelDirOption = new Option<string>("--model-dir")
{
    Description = "Directory to write image_encoder.onnx, tag_vocabulary.json and tag_embeddings.bin",
    DefaultValueFactory = _ => "./model"
};

var exportOnnxCommand = new Command("export-onnx", "Export a training checkpoint to the ONNX bundle unbooru-tagger-inference reads");
exportOnnxCommand.Options.Add(checkpointDirOption);
exportOnnxCommand.Options.Add(modelDirOption);
exportOnnxCommand.SetAction(parseResult => ExportOnnxCommandHandler.Run(
    parseResult.GetRequiredValue(checkpointDirOption),
    parseResult.GetRequiredValue(modelDirOption)));

var rootCommand = new RootCommand("unbooru-tagger training CLI")
{
    trainCommand,
    addTagCommand,
    exportOnnxCommand
};

return rootCommand.Parse(args).Invoke();
