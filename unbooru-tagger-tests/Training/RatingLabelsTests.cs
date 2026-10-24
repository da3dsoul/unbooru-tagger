using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Scoring;
using UnbooruTagger.Core.Vocabulary;
using UnbooruTagger.Training.Training;

namespace UnbooruTagger.Tests.Training;

public class RatingLabelsTests
{
    [Fact]
    public void Strictest_PicksTheMostExplicitKnownRating()
    {
        Assert.Equal(ContentRating.Explicit, ImageRatings.Strictest(new[] { "g", "e", "s" }.Select(ImageRatings.Parse)));
        Assert.Equal(ContentRating.Sensitive, ImageRatings.Strictest(new[] { "s", "bogus" }.Select(ImageRatings.Parse)));
        Assert.Equal(ContentRating.Unknown, ImageRatings.Strictest([]));
    }

    [Fact]
    public void SaveAndLoad_RoundTrip_AndRejectAMismatchedCount()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            ContentRating[] ratings = [ContentRating.General, ContentRating.Explicit, ContentRating.Unknown];
            ImageRatings.Save(directory, ratings);
            Assert.Equal(ratings, ImageRatings.Load(directory, 3));
            Assert.Throws<InvalidDataException>(() => ImageRatings.Load(directory, 4));
            Assert.Null(ImageRatings.Load(Path.Combine(directory, "missing"), 3));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Apply_AddsCumulativeRowsUpToEachImagesLevel()
    {
        var vocabulary = TagVocabulary.CreateEmpty();
        vocabulary.AddTag("hat");
        var rows = RatingLabels.EnsureRows(vocabulary);
        Assert.Equal([1, 2, 3], rows);
        Assert.Equal(rows, RatingLabels.EnsureRows(vocabulary)); // idempotent

        var applied = RatingLabels.Apply(
            [[0], [0], [], []],
            [ContentRating.General, ContentRating.Questionable, ContentRating.Explicit, ContentRating.Unknown],
            rows);

        Assert.Equal([0], applied[0]);
        Assert.Equal([0, 1, 2], applied[1]);
        Assert.Equal([1, 2, 3], applied[2]);
        Assert.Empty(applied[3]);
    }

    [Fact]
    public void Evaluate_PerfectScores_ProduceADiagonalConfusionMatrix()
    {
        // 1-d embeddings: an image's value is its rating level, and all three label rows share one
        // embedding, so every label's score rises with the level; full-recall cutoffs then land
        // exactly between levels.
        ContentRating[] ratings = [.. Enumerable.Range(0, 40).Select(i => (ContentRating)(i % 4))];
        var pooled = ratings.Select(r => (float)r).ToArray();
        float[] tagEmbeddings = [1f, 1f, 1f];
        var report = RatingEvaluator.Evaluate(pooled, 1, ratings, tagEmbeddings, [0, 1, 2], new LogitScaling(10f, -5f), [1.0, 1.0, 1.0]);

        Assert.Equal(40, report.Images);
        Assert.All(report.Recall, r => Assert.Equal(1.0, r, 6));
        Assert.Equal(0.0, report.ExplicitOrQuestionableCalledGeneral, 6);
    }
}
