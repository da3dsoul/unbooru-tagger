using Microsoft.Data.Sqlite;
using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Core.Dataset;

/// <summary>Rows processed so far, and the total the source cache holds.</summary>
public delegate void CompactionProgress(int processed, int total);

/// <summary>
/// A compaction's outcome: how many rows the source cache had vs. how many survived,
/// and the old-&gt;new row-index mapping every kept row shifted to -- a caller that
/// tracks cache rows by index outside this cache directory (e.g. crawl.sqlite's
/// <c>Images.CacheRowIndex</c>) needs this to renumber its own references to match.
/// Any row NOT present as a key was dropped.
/// </summary>
public sealed record CompactionResult(int OriginalCount, int KeptCount, IReadOnlyDictionary<int, int> OldToNewIndex);

/// <summary>
/// Physically removes specific rows from a cache directory's <c>images.bin</c>/
/// <c>tag_rows.sqlite</c>, shifting every later surviving row down to close the gap --
/// for correcting a cache that ended up holding images it should never have kept (e.g.
/// video/GIF URLs that "successfully" decoded a single static frame despite carrying no
/// real still-image content, or rows orphaned by a past crash that lost their
/// crawl.sqlite tracking without the pixel data itself ever being cleaned up alongside
/// it -- <see cref="PreprocessedDatasetCache"/>'s own record format has no notion of
/// crawl.sqlite at all, so a row can go untracked there while still sitting in the
/// pixel cache forever).
///
/// A single sequential forward pass over the source cache, start to finish -- NOT
/// through <see cref="PreprocessedDatasetCacheReader"/>, deliberately: that type builds
/// a full random-access offset table (one box-header read per record) and loads every
/// tag row into memory up front, both O(original row count) walks that exist to support
/// <em>random</em> access training needs, neither of which this purely sequential,
/// forward-only rewrite needs at all. On a multi-million-row, multi-terabyte cache the
/// difference is the gap between millions of small positioned reads (confirmed for real:
/// 26 million read syscalls and no visible progress in under a minute, on an array
/// capable of 2GB/s sequential) and one continuous streamed copy.
///
/// Same in-place-via-temp-directory-then-swap strategy as
/// <see cref="PreprocessedDatasetCacheMigrator.ShrinkInPlace"/>: a crash or
/// cancellation partway through leaves the original cache completely untouched. UNLIKE
/// the migrator, this doesn't need a source-row-count-based resume check -- since
/// records are read strictly in order and <paramref name="rowsToDrop"/> is the same
/// fixed input on every call, exactly how many SOURCE rows have been consumed is
/// tracked in its own small progress sidecar (<see cref="ProgressFileName"/>, written
/// alongside every <see cref="PreprocessedDatasetCacheWriter.Flush"/>) so a resumed run
/// can seek straight back to where the last one left off instead of restarting the
/// whole walk.
/// </summary>
public static class PreprocessedDatasetCacheCompactor
{
    /// <summary>How often (in SOURCE rows processed, not kept rows written) progress is durably checkpointed and reported -- matches <see cref="PreprocessedDatasetCacheWriter"/>'s own append-checkpoint cadence elsewhere in this file, so a crash loses at most this many rows' worth of re-work on resume.</summary>
    private const int CheckpointIntervalRows = 2_000;

    private const string ProgressFileName = ".compact-progress";
    private const int ProgressMagic = 0x434D5031; // "CMP1"

    /// <summary>
    /// Rewrites <paramref name="directory"/>'s cache with every index in
    /// <paramref name="rowsToDrop"/> removed. An empty/no-op <paramref name="rowsToDrop"/>
    /// is rejected -- callers should skip calling this entirely rather than pay for a
    /// full-cache copy that changes nothing. <paramref name="rowsToDrop"/> MUST be
    /// exactly the same set on every call against the same in-progress temp directory --
    /// a resumed run trusts it, unchecked, to reconstruct the already-processed prefix
    /// of <see cref="CompactionResult.OldToNewIndex"/> purely by replaying the same
    /// keep/drop decision per index rather than re-deriving it from disk.
    /// </summary>
    public static CompactionResult DropRows(
        string directory,
        IReadOnlySet<int> rowsToDrop,
        CompactionProgress? onProgress = null,
        CancellationToken cancellationToken = default)
    {
        if (rowsToDrop.Count == 0)
            throw new ArgumentException("Nothing to drop -- caller should skip compaction entirely instead of calling this with an empty set.", nameof(rowsToDrop));

        var pixelPath = Path.Combine(directory, PreprocessedDatasetCache.PixelsFileName);
        int originalCount;
        int inputSize;
        using (var headerStream = File.OpenRead(pixelPath))
        using (var headerReader = new BinaryReader(headerStream))
        {
            var magic = headerReader.ReadInt32();
            if (magic != PreprocessedDatasetCache.FormatMagic)
                throw new InvalidDataException($"Cache at '{directory}' was built with an older cache format. Delete it and rebuild.");

            originalCount = headerReader.ReadInt32();
            inputSize = headerReader.ReadInt32();
        }

        var tempDirectory = Path.Combine(directory, ".compact-tmp");
        var progressPath = Path.Combine(tempDirectory, ProgressFileName);

        var sourceRowsProcessed = 0;
        var sourceByteOffset = (long)PreprocessedDatasetCache.HeaderBytes;
        if (Directory.Exists(tempDirectory))
        {
            if (TryReadProgress(progressPath, out var resumeRows, out var resumeOffset))
            {
                sourceRowsProcessed = resumeRows;
                sourceByteOffset = resumeOffset;
            }
            else
            {
                Directory.Delete(tempDirectory, recursive: true); // no usable resume marker -- start over rather than guess
            }
        }

        Directory.CreateDirectory(tempDirectory);

        // Deterministically replay the keep/drop decision for every source row already
        // consumed by a prior (interrupted) run -- cheap pure in-memory set lookups, no
        // I/O -- to reconstruct the prefix of OldToNewIndex a resumed run can't recover
        // from the temp cache itself (that only tracks pixel/tag-row bytes, not the
        // original->compacted index mapping).
        var oldToNew = new Dictionary<int, int>(originalCount - rowsToDrop.Count);
        var keptSoFar = 0;
        for (var i = 0; i < sourceRowsProcessed; i++)
        {
            if (!rowsToDrop.Contains(i))
            {
                oldToNew[i] = keptSoFar;
                keptSoFar++;
            }
        }

        // Every one of these must be closed BEFORE SwapIn renames both the original and
        // temp copies of tag_rows.sqlite out from under whatever connection is holding
        // them open -- writer's connection targets the TEMP file SwapIn is about to move
        // into place, and tagRowStore's targets the ORIGINAL file SwapIn is about to
        // move aside, so both need to be done, not just idle, first.
        using (var writer = PreprocessedDatasetCacheWriter.OpenOrCreate(tempDirectory, inputSize))
        {
            if (writer.ImageCount != keptSoFar)
                throw new InvalidDataException(
                    $"Resumed compaction at '{tempDirectory}' holds {writer.ImageCount} row(s) but the progress marker expects {keptSoFar} -- " +
                    "refusing to continue against mismatched/inconsistent state. Delete that directory to restart the compaction from scratch.");

            using var tagRowStore = TagRowStore.OpenForReading(directory, originalCount);
            using var tagRowEnumerator = tagRowStore.StreamAll(originalCount, sourceRowsProcessed).GetEnumerator();

            using var sourceStream = new FileStream(pixelPath, FileMode.Open, FileAccess.Read, FileShare.Read, SequentialScanBufferBytes, FileOptions.SequentialScan);
            using var sourceReader = new BinaryReader(sourceStream);
            sourceStream.Position = sourceByteOffset;

            for (var i = sourceRowsProcessed; i < originalCount; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var x = sourceReader.ReadInt32();
                var y = sourceReader.ReadInt32();
                var width = sourceReader.ReadInt32();
                var height = sourceReader.ReadInt32();
                var pixels = sourceReader.ReadBytes((int)PreprocessedDatasetCache.ContentByteLength(width, height));

                if (!tagRowEnumerator.MoveNext() || tagRowEnumerator.Current.Index != i)
                    throw new InvalidDataException($"'{TagRowStore.DatabaseFileName}' is missing an entry for row {i} -- cache is inconsistent, refusing to compact.");

                if (!rowsToDrop.Contains(i))
                {
                    var encoded = new EncodedImage(pixels, new LetterboxBox(x, y, width, height));
                    writer.Append(encoded, tagRowEnumerator.Current.TagRows);
                    oldToNew[i] = writer.ImageCount - 1;
                }

                var rowsProcessed = i + 1;
                if (rowsProcessed % CheckpointIntervalRows == 0 || rowsProcessed == originalCount)
                {
                    writer.Flush();
                    WriteProgress(progressPath, rowsProcessed, sourceStream.Position);
                    onProgress?.Invoke(rowsProcessed, originalCount);
                }
            }
        }

        var keptCount = originalCount - rowsToDrop.Count;
        SwapIn(directory, tempDirectory, keptCount);
        return new CompactionResult(originalCount, keptCount, oldToNew);
    }

    /// <summary>Large enough that a sequential scan issues one real read syscall per several hundred/thousand records instead of one per record -- see <see cref="DetectValidPrefixLength"/>'s own doc comment for why that distinction matters at this cache's scale.</summary>
    private const int SequentialScanBufferBytes = 4 * 1024 * 1024;

    /// <summary>
    /// The number of LEADING rows in <paramref name="directory"/>'s <c>images.bin</c>
    /// whose box header decodes as a real (<c>width &gt; 0 &amp;&amp; height &gt; 0</c>)
    /// record, stopping at the first row that doesn't. A cache written correctly has
    /// every row valid, so this normally just returns the full row count; it exists for
    /// the failure case, where a write silently stopped landing real bytes partway
    /// through -- confirmed to happen for real once already: a compaction run reported
    /// success and the right final row count, but ~49% of those rows turned out to be
    /// <c>0x0</c> zero-content records, because verification at the time only checked
    /// row counts, never that content had actually been written.
    ///
    /// Reads every pixel payload into a discarded buffer rather than <see cref="Stream.Seek"/>-ing
    /// past it -- confirmed for real that the more "obviously" efficient-looking seek
    /// approach is dramatically WORSE here: each <c>Seek</c> invalidates a
    /// <see cref="FileStream"/>'s internal buffer, so the next record's 16-byte header
    /// read forces a fresh small refill from disk -- effectively one syscall per record
    /// again, the exact anti-pattern this whole class exists to avoid (see
    /// <see cref="DropRows"/>'s own doc comment: 26 million syscalls, no throughput, on
    /// an array capable of 2GB/s). Reading the payload keeps the stream's buffer full
    /// and the access pattern genuinely sequential, at the cost of transferring bytes
    /// this method never inspects -- still far cheaper than the seek pattern in
    /// practice, and this never decodes/validates those bytes as pixels (that's what
    /// makes this fast -- see the class's own "guarantee" caveat below).
    ///
    /// Not a guarantee the surviving pixel bytes are otherwise undamaged -- only that
    /// every row's box header looks like a real record.
    /// </summary>
    public static int DetectValidPrefixLength(string directory)
    {
        var pixelPath = Path.Combine(directory, PreprocessedDatasetCache.PixelsFileName);
        using var stream = new FileStream(pixelPath, FileMode.Open, FileAccess.Read, FileShare.Read, SequentialScanBufferBytes, FileOptions.SequentialScan);
        using var reader = new BinaryReader(stream);

        var magic = reader.ReadInt32();
        if (magic != PreprocessedDatasetCache.FormatMagic)
            throw new InvalidDataException($"Cache at '{directory}' was built with an older cache format. Delete it and rebuild.");
        var count = reader.ReadInt32();
        reader.ReadInt32(); // inputSize -- not needed here

        byte[]? discardBuffer = null;
        for (var i = 0; i < count; i++)
        {
            reader.ReadInt32(); // x
            reader.ReadInt32(); // y
            var width = reader.ReadInt32();
            var height = reader.ReadInt32();
            if (width <= 0 || height <= 0)
                return i;

            var contentLength = (int)PreprocessedDatasetCache.ContentByteLength(width, height);
            if (discardBuffer is null || discardBuffer.Length < contentLength)
                discardBuffer = new byte[contentLength];
            stream.ReadExactly(discardBuffer, 0, contentLength);
        }

        return count;
    }

    /// <summary>
    /// Copies the first <paramref name="rowCount"/> rows of <paramref name="sourceDirectory"/>'s
    /// cache into a brand new cache at <paramref name="destinationDirectory"/> -- unlike
    /// <see cref="DropRows"/>, this never touches <paramref name="sourceDirectory"/>
    /// beyond reading it, since the whole point is recovering INTO a fresh location when
    /// the source itself might be damaged past <paramref name="rowCount"/> (see
    /// <see cref="DetectValidPrefixLength"/>) and must be left alone for further
    /// inspection/recovery rather than risk compounding the damage with another in-place
    /// rewrite.
    ///
    /// No fine-grained resume: a interrupted/cancelled run's partial destination is
    /// discarded and the whole copy restarts next time, rather than paying for a
    /// progress sidecar the way <see cref="DropRows"/> does -- reasonable here because
    /// this is a plain sequential byte copy with no per-row decode/re-encode or
    /// network I/O, so even a multi-million-row prefix finishes in well under an hour;
    /// restarting it costs minutes, not the hours/days a network-bound rebuild of the
    /// REST of a cache would. A destination that already holds AT LEAST
    /// <paramref name="rowCount"/> rows is treated as done and left alone -- not just
    /// an exact-match check: a caller (dataset recovery) that appends MORE rows onto
    /// this same destination afterward (e.g. re-downloading everything past the valid
    /// prefix) would otherwise look "incomplete" on any later re-run and get its
    /// destination wiped and restarted from nothing, destroying however much of that
    /// far more expensive, network-bound follow-on work had already landed. Confirmed
    /// this would have happened for real: caught it about to fire against a
    /// destination that already had a further ~110,000 downloaded rows past the
    /// prefix, moments before a routine restart would have deleted all of it.
    /// </summary>
    public static void CopyPrefix(
        string sourceDirectory,
        string destinationDirectory,
        int rowCount,
        CompactionProgress? onProgress = null,
        CancellationToken cancellationToken = default)
    {
        var sourcePixelPath = Path.Combine(sourceDirectory, PreprocessedDatasetCache.PixelsFileName);
        int inputSize;
        using (var headerStream = File.OpenRead(sourcePixelPath))
        using (var headerReader = new BinaryReader(headerStream))
        {
            var magic = headerReader.ReadInt32();
            if (magic != PreprocessedDatasetCache.FormatMagic)
                throw new InvalidDataException($"Cache at '{sourceDirectory}' was built with an older cache format. Delete it and rebuild.");
            headerReader.ReadInt32(); // source's own row count -- irrelevant, rowCount is the caller's explicit request
            inputSize = headerReader.ReadInt32();
        }

        Directory.CreateDirectory(destinationDirectory);
        var destinationPixelPath = Path.Combine(destinationDirectory, PreprocessedDatasetCache.PixelsFileName);
        if (File.Exists(destinationPixelPath))
        {
            using var existingStream = File.OpenRead(destinationPixelPath);
            using var existingReader = new BinaryReader(existingStream);
            existingReader.ReadInt32();
            var existingCount = existingReader.ReadInt32();
            if (existingCount >= rowCount)
            {
                onProgress?.Invoke(rowCount, rowCount);
                return; // already fully copied (and possibly grown further by a caller appending afterward) -- see doc comment
            }
        }

        Directory.Delete(destinationDirectory, recursive: true);
        Directory.CreateDirectory(destinationDirectory);

        using var writer = new PreprocessedDatasetCacheWriter(destinationDirectory, inputSize);
        using var tagRowStore = TagRowStore.OpenForReading(sourceDirectory, rowCount);
        using var tagRowEnumerator = tagRowStore.StreamAll(rowCount).GetEnumerator();

        using var sourceStream = new FileStream(sourcePixelPath, FileMode.Open, FileAccess.Read, FileShare.Read, SequentialScanBufferBytes, FileOptions.SequentialScan);
        using var sourceReader = new BinaryReader(sourceStream);
        sourceStream.Position = PreprocessedDatasetCache.HeaderBytes;

        for (var i = 0; i < rowCount; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var x = sourceReader.ReadInt32();
            var y = sourceReader.ReadInt32();
            var width = sourceReader.ReadInt32();
            var height = sourceReader.ReadInt32();
            var pixels = sourceReader.ReadBytes((int)PreprocessedDatasetCache.ContentByteLength(width, height));

            if (!tagRowEnumerator.MoveNext() || tagRowEnumerator.Current.Index != i)
                throw new InvalidDataException($"'{TagRowStore.DatabaseFileName}' is missing an entry for row {i} -- cache is inconsistent, refusing to copy.");

            writer.Append(new EncodedImage(pixels, new LetterboxBox(x, y, width, height)), tagRowEnumerator.Current.TagRows);

            if ((i + 1) % CheckpointIntervalRows == 0 || i + 1 == rowCount)
            {
                writer.Flush();
                onProgress?.Invoke(i + 1, rowCount);
            }
        }
    }

    /// <summary>
    /// Streams a cache directory's tag rows in index order, without also opening/paying
    /// for its <c>images.bin</c> at all -- <see cref="TagRowStore"/> is internal to this
    /// assembly, so this is the public entry point a caller outside it (e.g. a dataset
    /// recovery tool trusting an already-compacted cache's tag rows independently of a
    /// separate pixel-content problem in the same directory) has to reach it through.
    /// </summary>
    public static IEnumerable<(int Index, IReadOnlyList<int> TagRows)> StreamTagRows(string directory, int rowCount, int startAtIndex = 0)
    {
        using var store = TagRowStore.OpenForReading(directory, rowCount);
        foreach (var row in store.StreamAll(rowCount, startAtIndex))
            yield return row;
    }

    private static void WriteProgress(string path, long sourceRowsProcessed, long sourceByteOffset)
    {
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(stream);
        writer.Write(ProgressMagic);
        writer.Write(sourceRowsProcessed);
        writer.Write(sourceByteOffset);
    }

    private static bool TryReadProgress(string path, out int sourceRowsProcessed, out long sourceByteOffset)
    {
        sourceRowsProcessed = 0;
        sourceByteOffset = 0;
        if (!File.Exists(path))
            return false;

        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var reader = new BinaryReader(stream);
            if (reader.ReadInt32() != ProgressMagic)
                return false;

            sourceRowsProcessed = checked((int)reader.ReadInt64());
            sourceByteOffset = reader.ReadInt64();
            return true;
        }
        catch (IOException)
        {
            // Covers EndOfStreamException (a truncated/corrupt marker) too -- it derives from IOException.
            return false;
        }
    }

    private static void SwapIn(string directory, string tempDirectory, int expectedImageCount)
    {
        var pixelPath = Path.Combine(directory, PreprocessedDatasetCache.PixelsFileName);
        var tagRowDbPath = Path.Combine(directory, TagRowStore.DatabaseFileName);
        var resumeIndexPath = Path.Combine(directory, PreprocessedDatasetCache.ResumeIndexFileName);
        var backupPixelPath = pixelPath + ".pre-compact.bak";
        var backupTagRowPath = tagRowDbPath + ".pre-compact.bak";

        // Same-volume renames (temp dir is a subdirectory of the cache directory being
        // compacted), so this is near-instant even for a multi-hundred-GB pixel file --
        // no bytes actually move here, only the swap at the very end (the two
        // File.Delete calls of the backups) touches that much data again.
        File.Move(pixelPath, backupPixelPath, overwrite: true);
        File.Move(tagRowDbPath, backupTagRowPath, overwrite: true);
        File.Move(Path.Combine(tempDirectory, PreprocessedDatasetCache.PixelsFileName), pixelPath);
        File.Move(Path.Combine(tempDirectory, TagRowStore.DatabaseFileName), tagRowDbPath);

        // Microsoft.Data.Sqlite pools native connections by connection string (which
        // embeds the file path) by default. This method's caller already opened and
        // closed at least one connection against tagRowDbPath's OLD content before this
        // rename -- if that connection went back into the pool instead of truly closing,
        // the verification read below (and any later caller in this same process) could
        // get handed that stale pooled connection instead of a fresh open against the
        // file this rename just swapped in, silently reading the pre-compaction rows.
        // Confirmed for real: without this, a same-process reader immediately after
        // SwapIn read back the ORIGINAL (uncompacted) tag rows despite the pixel file
        // compacting correctly.
        SqliteConnection.ClearAllPools();

        var tempResumeIndexPath = Path.Combine(tempDirectory, PreprocessedDatasetCache.ResumeIndexFileName);
        if (File.Exists(tempResumeIndexPath))
            File.Move(tempResumeIndexPath, resumeIndexPath, overwrite: true);
        else if (File.Exists(resumeIndexPath))
            File.Delete(resumeIndexPath); // stale -- refers to the pre-compaction pixel file's offsets

        // The progress sidecar's job ends here -- everything it tracked is now durably
        // reflected in the swapped-in files themselves.
        var tempProgressPath = Path.Combine(tempDirectory, ProgressFileName);
        if (File.Exists(tempProgressPath))
            File.Delete(tempProgressPath);

        // Non-recursive: anything unexpected still sitting in tempDirectory at this
        // point (the files above are everything this compaction ever creates) means
        // something's wrong, and should throw rather than silently vanish via a
        // recursive delete.
        Directory.Delete(tempDirectory);

        // Deliberately NOT a PreprocessedDatasetCacheReader open -- that builds the same
        // full random-access offset table and loads every tag row this whole class
        // exists to avoid paying for. DetectValidPrefixLength (header-only, seeks PAST
        // every pixel payload rather than reading it) plus an index-only SQLite
        // COUNT(*) (never touches the BLOB column) verify the swap without
        // re-introducing the exact cost this rewrite fixed.
        //
        // A row-COUNT check alone used to be all this did, and it isn't enough: a real
        // run once reported the exactly-correct final row count while ~49% of those
        // rows were silently written as empty 0x0 records instead of real pixel
        // content -- a write-side failure downstream of this method that a count-only
        // check has no way to see. DetectValidPrefixLength catches that shape of
        // failure specifically (every row genuinely has content, not just the right
        // number of them) without paying to read the content itself.
        var verifiedValidPrefix = DetectValidPrefixLength(directory);

        long verifiedTagRowCount;
        using (var connection = new SqliteConnection($"Data Source={tagRowDbPath}"))
        {
            connection.Open();
            using var count = connection.CreateCommand();
            count.CommandText = "SELECT COUNT(*) FROM TagRows;";
            verifiedTagRowCount = (long)count.ExecuteScalar()!;
        }

        if (verifiedValidPrefix != expectedImageCount || verifiedTagRowCount != expectedImageCount)
            throw new InvalidDataException(
                $"Compacted cache at '{directory}' has {verifiedValidPrefix} row(s) with real content (of which {verifiedTagRowCount} tag row(s) total), expected {expectedImageCount} -- " +
                $"leaving '{backupPixelPath}' and '{backupTagRowPath}' in place; do not delete them.");

        File.Delete(backupPixelPath);
        File.Delete(backupTagRowPath);
    }
}
