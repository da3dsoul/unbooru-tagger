using System.Text.Json;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Tests.Core;

public class TagRowMigrationTests
{
    private static EncodedImage Image(int inputSize, byte offset) =>
        new(
            Enumerable.Range(0, 3 * inputSize * inputSize).Select(i => (byte)(i + offset)).ToArray(),
            new LetterboxBox(0, 0, inputSize, inputSize));

    /// <summary>
    /// Rewrites an already-built current-format cache's tag storage back into the
    /// legacy <c>tag_rows.jsonl</c> format and deletes <c>tag_rows.sqlite</c> —
    /// simulating a cache built before <see cref="TagRowStore"/> existed (real caches
    /// like this exist: every one of this project's own crawler/build-large-cache/
    /// refresh-tags commands used to produce exactly this).
    /// </summary>
    private static void DowngradeToLegacyJsonl(string directory, IReadOnlyList<IReadOnlyList<int>> tagRows)
    {
        var jsonlPath = Path.Combine(directory, "tag_rows.jsonl");
        using (var writer = new StreamWriter(jsonlPath))
        {
            foreach (var row in tagRows)
                writer.WriteLine(JsonSerializer.Serialize(row));
        }

        File.Delete(Path.Combine(directory, "tag_rows.sqlite"));
    }

    [Fact]
    public void OpenForWriting_MigratesLegacyJsonlAutomatically_AndBacksUpTheOriginal()
    {
        const int inputSize = 2;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var tagRows = new List<IReadOnlyList<int>> { new[] { 0, 2 }, new[] { 1 } };
            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize))
            {
                writer.Append(Image(inputSize, 0), tagRows[0]);
                writer.Append(Image(inputSize, 50), tagRows[1]);
            }

            DowngradeToLegacyJsonl(directory, tagRows);
            Assert.True(File.Exists(Path.Combine(directory, "tag_rows.jsonl")));
            Assert.False(File.Exists(Path.Combine(directory, "tag_rows.sqlite")));

            var migrateCalls = new List<(string SubPhase, int Completed, int Total)>();
            using (var resumed = PreprocessedDatasetCacheWriter.OpenOrCreate(directory, inputSize,
                       onResumeProgress: (subPhase, completed, total) => migrateCalls.Add((subPhase, completed, total))))
            {
                Assert.Equal(2, resumed.ImageCount);
                var committed = resumed.ReadCommittedTagRows();
                Assert.Equal(new[] { 0, 2 }, committed[0]);
                Assert.Equal(new[] { 1 }, committed[1]);
            }

            Assert.Contains(migrateCalls, c => c.SubPhase == "migrating tag rows to SQLite");
            Assert.True(File.Exists(Path.Combine(directory, "tag_rows.sqlite")));
            Assert.True(File.Exists(Path.Combine(directory, "tag_rows.jsonl.migrated.bak")));
            Assert.False(File.Exists(Path.Combine(directory, "tag_rows.jsonl")));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void OpenForReading_MigratesLegacyJsonlAutomatically()
    {
        const int inputSize = 2;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var tagRows = new List<IReadOnlyList<int>> { new[] { 3 }, new[] { 4, 5 } };
            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize))
            {
                writer.Append(Image(inputSize, 0), tagRows[0]);
                writer.Append(Image(inputSize, 50), tagRows[1]);
            }

            DowngradeToLegacyJsonl(directory, tagRows);

            using var reader = new PreprocessedDatasetCacheReader(directory);
            Assert.Equal(new[] { 3 }, reader.ImageTagRows[0]);
            Assert.Equal(new[] { 4, 5 }, reader.ImageTagRows[1]);
            Assert.True(File.Exists(Path.Combine(directory, "tag_rows.sqlite")));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Migration_IgnoresDanglingLinesPastImageCount()
    {
        // A jsonl file can have trailing lines beyond ImageCount if the OLD writer
        // crashed mid-page before this migration ever existed -- migration must only
        // import the first ImageCount confirmed rows, same as the old writer's own
        // dangling-line handling used to.
        const int inputSize = 2;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize))
                writer.Append(Image(inputSize, 0), [0]);

            var jsonlPath = Path.Combine(directory, "tag_rows.jsonl");
            using (var jsonlWriter = new StreamWriter(jsonlPath))
            {
                jsonlWriter.WriteLine(JsonSerializer.Serialize(new[] { 0 }));
                jsonlWriter.WriteLine(JsonSerializer.Serialize(new[] { 99 })); // dangling -- past ImageCount == 1
            }

            File.Delete(Path.Combine(directory, "tag_rows.sqlite"));

            using var reader = new PreprocessedDatasetCacheReader(directory);
            Assert.Equal(1, reader.ImageCount);
            Assert.Equal(new[] { 0 }, reader.ImageTagRows[0]);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void Migration_RecoversFromAnInterruptedPriorAttempt()
    {
        // Simulates a crash partway through a PRIOR migration attempt: a leftover
        // ".migrating" temp database sitting next to the still-untouched original
        // jsonl. The next open must discard the stale temp file and migrate cleanly
        // from scratch, not treat the half-built temp file as real.
        const int inputSize = 2;
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            var tagRows = new List<IReadOnlyList<int>> { new[] { 7 } };
            using (var writer = new PreprocessedDatasetCacheWriter(directory, inputSize))
                writer.Append(Image(inputSize, 0), tagRows[0]);

            DowngradeToLegacyJsonl(directory, tagRows);

            File.WriteAllText(Path.Combine(directory, "tag_rows.sqlite.migrating"), "not a real sqlite file");

            using var reader = new PreprocessedDatasetCacheReader(directory);
            Assert.Equal(1, reader.ImageCount);
            Assert.Equal(new[] { 7 }, reader.ImageTagRows[0]);
            Assert.False(File.Exists(Path.Combine(directory, "tag_rows.sqlite.migrating")));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
