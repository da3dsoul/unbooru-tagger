using System.Text.Json;
using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Core.Dataset;

/// <summary>
/// Reads a cache written under the old fixed-stride, full-padded-canvas, float32
/// "LBX1" format — the layout <see cref="PreprocessedDatasetCache"/> used before
/// switching to variable-length, content-only, uint8 records ("LBX2"). Exists solely
/// so <see cref="PreprocessedDatasetCacheMigrator"/> can shrink an old cache in place;
/// nothing else should ever need to read this format again.
/// </summary>
internal sealed class LegacyPreprocessedDatasetCacheReader : IDisposable
{
    private const int LegacyFormatMagic = 0x4C425831; // "LBX1"

    private readonly FileStream _pixelStream;
    private readonly int _imageFloats;
    private readonly int _imageBytes;

    public int ImageCount { get; }
    public int InputSize { get; }

    public LegacyPreprocessedDatasetCacheReader(string directory)
    {
        _pixelStream = File.OpenRead(Path.Combine(directory, PreprocessedDatasetCache.PixelsFileName));
        using (var reader = new BinaryReader(_pixelStream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            var magic = reader.ReadInt32();
            if (magic != LegacyFormatMagic)
                throw new InvalidDataException(
                    $"Cache at '{directory}' isn't in the old \"LBX1\" format — nothing to migrate (already migrated, or a different format entirely).");

            ImageCount = reader.ReadInt32();
            InputSize = reader.ReadInt32();
        }

        _imageFloats = 3 * InputSize * InputSize;
        _imageBytes = PreprocessedDatasetCache.BoxBytes + (_imageFloats * sizeof(float));
    }

    public PreprocessedImage ReadImage(int index)
    {
        _pixelStream.Position = PreprocessedDatasetCache.HeaderBytes + ((long)index * _imageBytes);

        var buffer = new byte[_imageBytes];
        var read = _pixelStream.Read(buffer, 0, buffer.Length);
        if (read != buffer.Length)
            throw new EndOfStreamException($"Expected {buffer.Length} bytes for image {index}, got {read}.");

        var content = new LetterboxBox(
            BitConverter.ToInt32(buffer, 0),
            BitConverter.ToInt32(buffer, 4),
            BitConverter.ToInt32(buffer, 8),
            BitConverter.ToInt32(buffer, 12));

        var floats = new float[_imageFloats];
        Buffer.BlockCopy(buffer, PreprocessedDatasetCache.BoxBytes, floats, 0, floats.Length * sizeof(float));
        return new PreprocessedImage(floats, content);
    }

    public void Dispose() => _pixelStream.Dispose();
}

/// <summary>Progress callback: images converted so far, and the total the source cache holds.</summary>
public delegate void ShrinkProgress(int converted, int total);

/// <summary>
/// One-time conversion of an old "LBX1"-format <see cref="PreprocessedDatasetCache"/>
/// directory to the current "LBX2" (content-only, uint8) format, in place. Works
/// entirely from the already-decoded/normalized pixels already sitting in the old
/// cache — it does not need (and the crawler never keeps) the original source images,
/// since every real pixel value is already captured there; this only strips the
/// redundant letterbox padding and drops float32 back down to the uint8 precision the
/// source images actually had.
/// </summary>
public static class PreprocessedDatasetCacheMigrator
{
    private const int FlushIntervalImages = 2000;

    /// <summary>
    /// Converts <paramref name="directory"/>'s cache to the current format. Writes the
    /// new files to a temporary subdirectory first and only swaps them in once every
    /// image has been converted and the result re-opens cleanly — a crash or
    /// cancellation partway through leaves the original cache untouched (beyond the
    /// resumable temp files) rather than corrupted. Re-running after an interruption
    /// resumes the temp conversion rather than restarting it.
    /// </summary>
    public static void ShrinkInPlace(string directory, ShrinkProgress? onProgress = null, CancellationToken cancellationToken = default)
    {
        using var legacyReader = new LegacyPreprocessedDatasetCacheReader(directory);

        var tagRowsPath = Path.Combine(directory, PreprocessedDatasetCache.LabelsFileName);
        var tagRows = File.ReadLines(tagRowsPath)
            .Take(legacyReader.ImageCount)
            .Select(line => (IReadOnlyList<int>)(JsonSerializer.Deserialize<int[]>(line)
                             ?? throw new InvalidDataException($"'{tagRowsPath}' contains an invalid tag-row label line.")))
            .ToList();
        if (tagRows.Count != legacyReader.ImageCount)
            throw new InvalidDataException(
                $"'{tagRowsPath}' has {tagRows.Count} committed row(s) but images.bin has {legacyReader.ImageCount} — refusing to migrate a cache whose two files disagree.");

        var tempDirectory = Path.Combine(directory, ".shrink-tmp");
        Directory.CreateDirectory(tempDirectory);

        using (var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(tempDirectory, legacyReader.InputSize))
        {
            for (var i = writer.ImageCount; i < legacyReader.ImageCount; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var legacyImage = legacyReader.ReadImage(i);
                var encoded = ImagePreprocessing.Strip(legacyImage, legacyReader.InputSize);
                writer.Append(encoded, tagRows[i]);

                if ((i + 1) % FlushIntervalImages == 0)
                    writer.Flush();

                onProgress?.Invoke(i + 1, legacyReader.ImageCount);
            }
        }

        SwapIn(directory, tempDirectory, legacyReader.ImageCount);
    }

    private static void SwapIn(string directory, string tempDirectory, int expectedImageCount)
    {
        var pixelPath = Path.Combine(directory, PreprocessedDatasetCache.PixelsFileName);
        var labelPath = Path.Combine(directory, PreprocessedDatasetCache.LabelsFileName); // the genuine LBX1-era tag_rows.jsonl being migrated away from
        var tagRowDbPath = Path.Combine(directory, TagRowStore.DatabaseFileName);
        var resumeIndexPath = Path.Combine(directory, PreprocessedDatasetCache.ResumeIndexFileName);
        var backupPixelPath = pixelPath + ".lbx1.bak";
        var backupLabelPath = labelPath + ".lbx1.bak";

        // Same-volume renames (temp dir is a subdirectory of the cache directory being
        // migrated), so this is near-instant even for a multi-hundred-GB pixel file —
        // no bytes actually move here, only the swap at the very end (File.Delete of
        // the backup) touches that much data again.
        File.Move(pixelPath, backupPixelPath, overwrite: true);
        File.Move(labelPath, backupLabelPath, overwrite: true);
        File.Move(Path.Combine(tempDirectory, PreprocessedDatasetCache.PixelsFileName), pixelPath);

        // The temp writer stores tag rows via TagRowStore (SQLite), not tag_rows.jsonl —
        // ShrinkInPlace only ever reads the OLD jsonl directly as migration source data
        // (see above), it never asks the temp writer to produce one.
        File.Move(Path.Combine(tempDirectory, TagRowStore.DatabaseFileName), tagRowDbPath, overwrite: true);

        // The temp writer's own Flush calls (see ShrinkInPlace) already wrote a valid
        // resume index for tempDirectory's cache — carry it over so the migrated cache
        // doesn't pay for one wasted walk on its very first resume. Not every migration
        // reaches this with one present (e.g. a source cache with zero images never
        // calls Flush), so this is a move, not a required file.
        var tempResumeIndexPath = Path.Combine(tempDirectory, PreprocessedDatasetCache.ResumeIndexFileName);
        if (File.Exists(tempResumeIndexPath))
            File.Move(tempResumeIndexPath, resumeIndexPath, overwrite: true);

        // Non-recursive: anything unexpected still sitting in tempDirectory at this
        // point (the two files and the resume index above are everything the temp
        // writer ever creates) means something's wrong, and should throw rather than
        // silently vanish via a recursive delete.
        Directory.Delete(tempDirectory);

        using (var verify = new PreprocessedDatasetCacheReader(directory))
        {
            if (verify.ImageCount != expectedImageCount)
                throw new InvalidDataException(
                    $"Migrated cache at '{directory}' reports {verify.ImageCount} images, expected {expectedImageCount} — " +
                    $"leaving '{backupPixelPath}' and '{backupLabelPath}' in place; do not delete them.");
        }

        File.Delete(backupPixelPath);
        File.Delete(backupLabelPath);
    }
}
