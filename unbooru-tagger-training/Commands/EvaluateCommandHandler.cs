using Spectre.Console;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Scoring;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Checkpoints;
using UnbooruTagger.Training.Model;
using UnbooruTagger.Training.Training;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Commands;

/// <summary>
/// Scores a checkpoint on the same held-out split its training run used (reconstructed
/// from <see cref="TrainingProgress.ValidationSplitSeed"/>), with
/// <see cref="TagRankingEvaluator"/>'s per-tag and image-level metrics, and writes the
/// calibrated per-tag thresholds plus a full per-tag report next to it — so any checkpoint
/// (including one trained before ranking evaluation existed) can be measured and get
/// thresholds without training further.
/// </summary>
public static class EvaluateCommandHandler
{
    public static int Run(
        string checkpointDir, string cacheDir, double validationFraction, int maxImages, int batchSize,
        string? exclusiveTagsPath, string? outDir, int worstTagsToShow, string? embeddingsCachePath = null,
        string? goldPath = null, int localizationSamples = 300)
    {
        var device = DeviceSelector.Best();
        using var cacheReader = new PreprocessedDatasetCacheReader(cacheDir);
        var (imageTower, config, vocabulary, embeddings, logitScaleBias) = Checkpoint.Load(checkpointDir, device);

        var datasetVocabulary = TagVocabulary.Load(Path.Combine(cacheDir, "tag_vocabulary.json"));

        // Mirror training: a checkpoint that learned ratings carries the three cumulative rating
        // rows, and each image gets them per its rating; then dataset rows are matched to the
        // checkpoint's (possibly compacted) vocabulary by tag name.
        var ratingRows = RatingLabels.FindRows(vocabulary);
        var ratings = ratingRows is not null ? ImageRatings.Load(cacheDir, cacheReader.ImageCount) : null;
        IReadOnlyList<IReadOnlyList<int>> datasetRows = cacheReader.ImageTagRows;
        var implications = TagImplications.Load(Path.Combine(cacheDir, TagImplications.FileName), datasetVocabulary);
        if (implications.RuleCount > 0)
            datasetRows = implications.Complete(datasetRows, out _, out _);
        if (ratingRows is not null && ratings is not null)
            datasetRows = RatingLabels.Apply(datasetRows, ratings, RatingLabels.EnsureRows(datasetVocabulary));
        var (remappedRows, unmatched) = VocabularyRemap.Apply(datasetVocabulary, datasetRows, vocabulary);
        if (unmatched > 0)
            AnsiConsole.MarkupLineInterpolated($"{unmatched} dataset tags aren't in the checkpoint's vocabulary and are ignored.");

        var seed = 0;
        if (TrainingState.HasProgress(checkpointDir))
            seed = TrainingState.LoadProgress(checkpointDir).ValidationSplitSeed;
        else
            AnsiConsole.MarkupLine("[yellow]No training_progress.json -- can't reconstruct this run's validation split; using seed 0, which may overlap its training images.[/]");

        IReadOnlyList<IReadOnlyList<int>> imageTagRows = remappedRows;
        var (trainingIndices, validationIndices) = TrainCommandHandler.SplitForValidation(cacheReader.ImageCount, validationFraction, seed);
        var heldOut = maxImages > 0 && validationIndices.Count > maxImages ? validationIndices.Take(maxImages).ToList() : validationIndices;

        AnsiConsole.MarkupLine("Building per-tag statistics from the training split...");
        var tagSites = TagAgreement.LoadSiteCoverage(Path.Combine(cacheDir, TagAgreement.FileName), vocabulary);
        var imageSites = tagSites is not null ? ImageSites.Load(cacheDir, cacheReader.ImageCount) : null;
        if (imageSites is null)
            tagSites = null;
        var statistics = TagStatistics.Build(trainingIndices.Select(i => imageTagRows[i]).ToList(), vocabulary.Records.Count, seed: seed,
            impliedThreshold: TrainingTargetOptions.Default.ImpliedThreshold, missingThreshold: TrainingTargetOptions.Default.MissingThreshold,
            imageSites: imageSites is not null ? trainingIndices.Select(i => imageSites[i]).ToArray() : null, tagSites: tagSites);
        var exclusiveGroups = ExclusiveTagGroups.Load(
            exclusiveTagsPath ?? Path.Combine(checkpointDir, ExclusiveTagGroups.FileName), vocabulary, fallBackToDefaults: true);

        var tagTower = TagTower.CreateFullyTrainable(
            Enumerable.Range(0, embeddings.RowCount).Select(i => embeddings.GetRow(i).ToArray()).ToArray(), config.EmbeddingDim, device);
        using var allTagIndices = tensor(Enumerable.Range(0, vocabulary.Records.Count).Select(i => (long)i).ToArray(), device: device);

        EvaluationReport? report = null;
        RatingReport? ratingReport = null;
        AnsiConsole.Progress()
            .Columns(new TaskDescriptionColumn(), new ProgressBarColumn(), new PercentageColumn(), new RemainingTimeColumn())
            .Start(ctx =>
            {
                var batches = Math.Max(1, (int)Math.Ceiling(heldOut.Count / (double)batchSize));
                var task = ctx.AddTask($"Embedding {heldOut.Count} held-out images", maxValue: batches);
                var pooled = TryLoadEmbeddings(embeddingsCachePath, heldOut.Count * config.EmbeddingDim, checkpointDir);
                if (pooled is not null)
                {
                    task.Value = batches;
                    AnsiConsole.MarkupLineInterpolated($"Reusing held-out embeddings from '{embeddingsCachePath}'.");
                }
                else
                {
                    (var loss, pooled) = TrainCommandHandler.Evaluate(
                        imageTower, tagTower, logitScaleBias,
                        indices => TrainCommandHandler.LoadCacheBatch(cacheReader, indices, cacheReader.InputSize),
                        imageTagRows, heldOut, allTagIndices, vocabulary.Records.Count, device, batchSize, cacheReader.InputSize,
                        onBatchComplete: done => task.Value = done,
                        readWindow: ids => cacheReader.ReadEncoded(ids));
                    AnsiConsole.MarkupLineInterpolated($"mean validation loss (old metric) {loss:G4}");
                    if (embeddingsCachePath is not null)
                        SaveEmbeddings(embeddingsCachePath, pooled);
                }

                var rankingTask = ctx.AddTask("Scoring tag rankings", maxValue: 1);
                report = TrainCommandHandler.EvaluateRankings(
                    pooled, heldOut.Select(i => imageTagRows[i]).ToList(), tagTower, allTagIndices, logitScaleBias,
                    vocabulary, statistics, exclusiveGroups, device, seed,
                    imageSites is not null ? heldOut.Select(i => imageSites[i]).ToArray() : null);
                rankingTask.Value = 1;

                if (ratingRows is not null && ratings is not null)
                    ratingReport = TrainCommandHandler.EvaluateRatings(
                        pooled, heldOut.Select(i => ratings[i]).ToList(), tagTower, allTagIndices, logitScaleBias, ratingRows, config.EmbeddingDim);
            });

        var outputDirectory = outDir ?? checkpointDir;
        Directory.CreateDirectory(outputDirectory);
        TrainCommandHandler.WriteEvaluationArtifacts(outputDirectory, report!, ratingReport);

        LocalizationReport? localization = null;
        if (localizationSamples > 0 || goldPath is not null)
        {
            AnsiConsole.MarkupLine("Measuring localization...");
            var tagEmbeddingArray = Enumerable.Range(0, embeddings.RowCount).SelectMany(i => embeddings.GetRow(i).ToArray()).ToArray();
            var evaluator = new LocalizationEvaluator(
                imageTower, tagEmbeddingArray, config.EmbeddingDim, logitScaleBias.ToLogitScaling(),
                ids => cacheReader.ReadEncoded(ids), cacheReader.InputSize, device);
            var gold = goldPath is not null ? LocalizationEvaluator.LoadGold(goldPath) : null;
            var localizability = TagLocalizability.Load(Path.Combine(checkpointDir, TagLocalizability.FileName));
            localization = evaluator.Evaluate(vocabulary, imageTagRows, heldOut, gold, localizability, Math.Max(localizationSamples, 1), seed);
            WriteLocalizationReport(outputDirectory, localization);
        }

        AnsiConsole.MarkupLine(TrainCommandHandler.DescribeReport(report!));
        if (ratingReport is not null)
            AnsiConsole.MarkupLine(TrainCommandHandler.DescribeRatings(ratingReport));
        else
            AnsiConsole.MarkupLine("[yellow]No rating evaluation: this checkpoint has no rating labels (trained before ratings existed) or the dataset has no ratings.bin.[/]");
        if (localization is not null)
        {
            AnsiConsole.MarkupLine(DescribeLocalization(localization));
            if (localization.GoldTags.Count > 0)
            {
                var table = new Table().Title("Gold-set localization by tag (worst pointing first)")
                    .AddColumns("tag", "pairs", "gold boxes", "pointing", "energy in box", "box precision", "box recall", "count error");
                foreach (var t in localization.GoldTags)
                    table.AddRow(Markup.Escape(t.Name), t.Pairs.ToString(), t.GoldInstances.ToString(), t.PointingAccuracy.ToString("F2"),
                        t.EnergyInBox.ToString("F2"), t.InstancePrecision.ToString("F2"), t.InstanceRecall.ToString("F2"), t.MeanCountError.ToString("F2"));
                AnsiConsole.Write(table);
            }

            foreach (var t in localization.Teacher ?? [])
                AnsiConsole.MarkupLine(DescribeTeacher(t));
        }
        PrintTags($"Worst {worstTagsToShow} tags by AP", report!.Tags.Take(worstTagsToShow));
        PrintTags($"Most-confused {worstTagsToShow} tags by hard-negative AP",
            report.Tags.Where(t => !double.IsNaN(t.HardAveragePrecision)).OrderBy(t => t.HardAveragePrecision).Take(worstTagsToShow));
        AnsiConsole.MarkupLineInterpolated(
            $"Wrote per-tag thresholds and the full report to '{outputDirectory}' ({Core.Scoring.TagThresholds.FileName}, {TrainCommandHandler.EvaluationReportFileName}).");
        return 0;
    }

    /// <summary>
    /// Held-out embeddings only depend on the image tower and the held-out image list, so a
    /// re-evaluation (different thresholds/exclusions/metrics) can skip the disk-bound
    /// embedding pass. Only reused if newer than the checkpoint's image tower and the right size.
    /// </summary>
    private static float[]? TryLoadEmbeddings(string? path, int expectedLength, string checkpointDir)
    {
        if (path is null || !File.Exists(path))
            return null;
        var tower = Path.Combine(checkpointDir, "image_tower.dat");
        if (File.Exists(tower) && File.GetLastWriteTimeUtc(tower) > File.GetLastWriteTimeUtc(path))
            return null;

        var bytes = File.ReadAllBytes(path);
        if (bytes.Length != expectedLength * sizeof(float))
            return null;
        var values = new float[expectedLength];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static void SaveEmbeddings(string path, float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
    }

    internal const string LocalizationReportFileName = "localization_report.json";

    internal static void WriteLocalizationReport(string directory, LocalizationReport report) =>
        File.WriteAllText(
            Path.Combine(directory, LocalizationReportFileName),
            System.Text.Json.JsonSerializer.Serialize(report, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true,
                // A rate with nothing to count (e.g. no exclude-crop fit an image) is NaN.
                NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals
            }));

    internal static string DescribeTeacher(TeacherResult t) =>
        Markup.Escape(
            $"crop teacher ({t.Name}): {t.IncludePairs} include / {t.ExcludePairs} exclude crops | full-image yes {t.FullImageYesRate:P1} | " +
            $"include-crop yes {t.IncludeYesRate:P1}, exclude-crop no {t.ExcludeNoRate:P1}, balanced {t.BalancedAccuracy:P1} | " +
            $"confident (p>0.9 or <0.1) {t.ConfidentCoverage:P1} of answers, {t.ConfidentAccuracy:P1} right");

    internal static string DescribeGold(GoldResult g) =>
        Markup.Escape(
            $"gold localization ({g.Name}, {g.Pairs} pairs): pointing {g.PointingAccuracy:P1}, energy in box {g.EnergyInBox:F3}, " +
            $"box precision {g.InstancePrecision:P2} / recall {g.InstanceRecall:P1}, count error {g.MeanCountError:F1}");

    internal static string DescribeLocalization(LocalizationReport r) =>
        Markup.Escape(
            string.Concat(r.Gold.Select(g =>
                $"gold {g.Name} ({g.Pairs} image-tag pairs): pointing {g.PointingAccuracy:P1}, energy in box {g.EnergyInBox:P1}, box precision {g.InstancePrecision:P1}, box recall {g.InstanceRecall:P1}, count error {g.MeanCountError:F2} | ")) +
            $"deletion gap {r.DeletionGap:F4} (top beats random {r.DeletionWinRate:P1}, {r.DeletionPairs} pairs) | " +
            $"two characters: heatmap correlation {r.TwoCharacterCorrelation:F3}, distinct peaks {r.TwoCharacterDistinctPeaks:P1} ({r.TwoCharacterImages} images) | " +
            $"crop consistency {r.CropConsistency:F3} ({r.CropPairs} pairs) | solo once-per-person tags with exactly one box {r.SoloSingleInstanceRate:P1} ({r.SoloPairs} pairs)");

    private static void PrintTags(string title, IEnumerable<TagEvaluation> tags)
    {
        var table = new Table().Title(Markup.Escape(title))
            .AddColumns("tag", "held-out positives", "hard negatives", "AP", "hard AP", "held-out F1", "threshold");
        foreach (var t in tags)
            table.AddRow(
                Markup.Escape(t.Tag), t.Positives.ToString(), t.HardNegatives.ToString(),
                t.AveragePrecision.ToString("F3"), double.IsNaN(t.HardAveragePrecision) ? "-" : t.HardAveragePrecision.ToString("F3"),
                double.IsNaN(t.HeldOutF1) ? "-" : t.HeldOutF1.ToString("F3"), t.Threshold.ToString("F3"));
        AnsiConsole.Write(table);
    }
}
