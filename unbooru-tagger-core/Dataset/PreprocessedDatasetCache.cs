using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Core.Dataset;

/// <summary>
/// A preprocessed image-tensor cache for large training runs: images are decoded,
/// resized, and normalized once during data prep instead of on every epoch. Pixel
/// data is written/read incrementally (streamed to/from disk, one image at a time)
/// so a large corpus never needs to fit in memory during preprocessing or training.
/// </summary>
public static class PreprocessedDatasetCache
{
    internal const string PixelsFileName = "images.bin";

    /// <summary>
    /// The legacy per-image tag-row label format — one JSON array per line — superseded
    /// by <see cref="TagRowStore"/> (a SQLite database, <c>tag_rows.sqlite</c>). Kept as
    /// a constant purely because two migrations still need to recognize/read it: an
    /// existing cache that predates <see cref="TagRowStore"/> gets migrated to it
    /// automatically the first time it's opened (see <see cref="TagRowStore.OpenForWriting"/>/
    /// <see cref="TagRowStore.OpenForReading"/>), and <see cref="PreprocessedDatasetCacheMigrator"/>
    /// (the much older LBX1 -&gt; LBX2 pixel-format migration) reads a genuinely ancient
    /// cache's copy of this file directly as its source data.
    /// </summary>
    internal const string LabelsFileName = "tag_rows.jsonl";

    /// <summary>
    /// Small sidecar caching, for the current <c>ImageCount</c>, the byte offset
    /// <see cref="PreprocessedImageIndex.FindEndOffset"/> would otherwise have to
    /// re-walk every already-cached pixel record's box header to find. See
    /// <see cref="PreprocessedDatasetCacheWriter.Flush"/> (writes it) and the writer's
    /// resume constructor (reads it). Deliberately a separate tiny file rather than a
    /// field inside <see cref="PixelsFileName"/>'s own header: growing that header
    /// would shift every existing record's byte offset, which would need rewriting the
    /// entire (potentially multi-GB) pixel file to adopt on an existing cache — exactly
    /// the expensive one-time cost this is meant to avoid. Self-healing: if it's
    /// missing, corrupt, or its stored count doesn't match the pixel file's current
    /// <c>ImageCount</c> (stale — e.g. from before this file existed, or a crash
    /// between the two writes), the reader falls back to the original walk and
    /// rewrites this file with the now-current answer.
    /// </summary>
    internal const string ResumeIndexFileName = "images.bin.resume";

    internal const int ResumeIndexMagic = 0x52534D31; // "RSM1"

    /// <summary>
    /// Distinguishes the current format from older ones — an old cache read under a
    /// newer layout would otherwise silently misinterpret its bytes instead of failing
    /// loudly. Bumped from "LBX1" to "LBX2" when records switched from fixed-size
    /// (full padded canvas, float32) to variable-size (content-only, uint8): a cache
    /// built under "LBX1" must be regenerated, not just re-read, since the two formats
    /// don't even agree on how long a record is.
    /// </summary>
    internal const int FormatMagic = 0x4C425832; // "LBX2"

    internal const int HeaderBytes = sizeof(int) * 3;
    internal const int BoxBytes = sizeof(int) * 4;

    /// <summary>
    /// Byte length of a record's pixel payload — <see cref="EncodedImage"/> stores only
    /// the letterbox content region (no padding), as raw <c>uint8</c> RGB, so this is
    /// derivable from the box already written at the start of the record instead of
    /// needing a separate stored length.
    /// </summary>
    internal static long ContentByteLength(int width, int height) => (long)width * height * 3;
}

/// <summary>
/// Locates records in <see cref="PreprocessedDatasetCache.PixelsFileName"/>, which are
/// variable-length (an <see cref="EncodedImage"/> stores only its letterbox content
/// region, no padding) rather than fixed-stride — a record's start offset can only be
/// found by walking every prior record's 16-byte box header (never its pixel payload,
/// which is what makes the walk cheap even across millions of records) and accumulating
/// <c>BoxBytes + width*height*3</c> each time.
/// </summary>
internal static class PreprocessedImageIndex
{
    /// <summary>How often (in records) a walk reports back to an <c>onProgress</c> callback — frequent enough to look live on a multi-million-record corpus, infrequent enough that the callback itself is never the bottleneck.</summary>
    private const int ProgressReportInterval = 5_000;

    /// <summary>The byte offset each of <paramref name="imageCount"/> records starts at, in image-index order — for building a reader's random-access index.</summary>
    public static long[] BuildOffsets(Stream stream, int imageCount)
    {
        var offsets = new long[imageCount];
        var position = (long)PreprocessedDatasetCache.HeaderBytes;
        var box = new byte[PreprocessedDatasetCache.BoxBytes];
        for (var i = 0; i < imageCount; i++)
        {
            offsets[i] = position;
            position += ReadRecordLength(stream, position, box, i);
        }

        return offsets;
    }

    /// <summary>
    /// The byte offset just past the last of <paramref name="imageCount"/> records —
    /// where a resumed writer should continue appending. One seek-past-the-payload +
    /// one 16-byte header read per record; storage with high per-seek latency (a
    /// network mount, a spinning disk) makes that add up visibly at a multi-million-
    /// record corpus, which is what <paramref name="onProgress"/> (called every
    /// <see cref="ProgressReportInterval"/> records, plus once at the end) is for.
    /// </summary>
    public static long FindEndOffset(Stream stream, int imageCount, Action<int, int>? onProgress = null)
    {
        var position = (long)PreprocessedDatasetCache.HeaderBytes;
        var box = new byte[PreprocessedDatasetCache.BoxBytes];
        for (var i = 0; i < imageCount; i++)
        {
            position += ReadRecordLength(stream, position, box, i);
            if (onProgress is not null && (i + 1) % ProgressReportInterval == 0)
                onProgress(i + 1, imageCount);
        }

        onProgress?.Invoke(imageCount, imageCount);
        return position;
    }

    private static long ReadRecordLength(Stream stream, long position, byte[] boxBuffer, int imageIndex)
    {
        stream.Position = position;
        var read = stream.Read(boxBuffer, 0, boxBuffer.Length);
        if (read != boxBuffer.Length)
            throw new EndOfStreamException($"Expected a {boxBuffer.Length}-byte box header for image {imageIndex}, got {read}.");

        var width = BitConverter.ToInt32(boxBuffer, 8);
        var height = BitConverter.ToInt32(boxBuffer, 12);
        return PreprocessedDatasetCache.BoxBytes + PreprocessedDatasetCache.ContentByteLength(width, height);
    }
}

/// <summary>
/// Appends to <see cref="PreprocessedDatasetCache"/>'s on-disk files: <c>images.bin</c>
/// for pixels, and a <see cref="TagRowStore"/> (SQLite) for each image's tag-row
/// indices. Tag rows used to be one JSON array per line in a <c>tag_rows.jsonl</c> text
/// file; <see cref="TagRowStore"/> migrates one of those automatically the first time
/// it's opened, so that history doesn't need repeating here.
/// </summary>
public sealed class PreprocessedDatasetCacheWriter : IDisposable
{
    private readonly FileStream _pixelStream;
    private readonly BinaryWriter _pixelWriter;
    private readonly string _resumeIndexPath;
    private readonly TagRowStore _tagRowStore;

    /// <summary>Images durably committed as of the last <see cref="Flush"/> (or resumed from a prior run).</summary>
    public int ImageCount { get; private set; }

    public PreprocessedDatasetCacheWriter(string directory, int inputSize)
        : this(directory, inputSize, resume: false)
    {
    }

    /// <summary>
    /// Opens an existing cache directory to continue appending where a prior run left
    /// off, or creates a fresh one if <paramref name="directory"/> is empty/missing.
    /// Any pixel bytes/tag rows left over from a page that started but never finished a
    /// <see cref="Flush"/> are dropped, so resumed appends start exactly at the last
    /// confirmed image boundary. <paramref name="onResumeProgress"/> (subPhase label,
    /// completed, total), if given, is invoked periodically while resuming against an
    /// existing corpus — both walking the pixel file's box headers (only needed when
    /// the resume-index sidecar is missing/stale) and a one-time <c>tag_rows.jsonl</c>
    /// migration (only needed once, ever, per cache) are O(<c>ImageCount</c>), so a
    /// multi-million-image corpus can spend real, visible time here with nothing else
    /// to show for it otherwise.
    /// </summary>
    public static PreprocessedDatasetCacheWriter OpenOrCreate(string directory, int inputSize, Action<string, int, int>? onResumeProgress = null) =>
        new(directory, inputSize, resume: true, onResumeProgress);

    private PreprocessedDatasetCacheWriter(string directory, int inputSize, bool resume, Action<string, int, int>? onResumeProgress = null)
    {
        Directory.CreateDirectory(directory);

        var pixelPath = Path.Combine(directory, PreprocessedDatasetCache.PixelsFileName);
        _resumeIndexPath = Path.Combine(directory, PreprocessedDatasetCache.ResumeIndexFileName);

        if (resume && File.Exists(pixelPath))
        {
            _pixelStream = new FileStream(pixelPath, FileMode.Open, FileAccess.ReadWrite);
            _pixelWriter = new BinaryWriter(_pixelStream);

            _pixelStream.Position = 0;
            using (var headerReader = new BinaryReader(_pixelStream, System.Text.Encoding.UTF8, leaveOpen: true))
            {
                var magic = headerReader.ReadInt32();
                if (magic != PreprocessedDatasetCache.FormatMagic)
                    throw new InvalidDataException(
                        $"Cache at '{directory}' was built with an older cache format. Delete it and rebuild.");

                ImageCount = headerReader.ReadInt32();
                var storedInputSize = headerReader.ReadInt32();
                if (storedInputSize != inputSize)
                    throw new InvalidDataException($"Cache at '{directory}' was built with input size {storedInputSize}, but {inputSize} was requested.");
            }

            // Records are variable-length (content-only, no padding stored), so the
            // resume point can't be computed by arithmetic the way a fixed-stride format
            // could — normally this means walking the ImageCount confirmed records' box
            // headers (16 bytes each, never the pixel payload) to find exactly where the
            // last one ends. The resume index (see ResumeIndexFileName) caches that
            // answer from the last Flush, so a clean resume skips the walk entirely; a
            // stale/missing/corrupt index is the only thing that falls back to doing it
            // the slow way, which then heals the index for next time.
            var hasValidResumeIndex = TryReadResumeIndex(_resumeIndexPath, ImageCount, out var pixelResumePosition);

            if (!hasValidResumeIndex)
            {
                pixelResumePosition = PreprocessedImageIndex.FindEndOffset(_pixelStream, ImageCount,
                    onProgress: (completed, total) => onResumeProgress?.Invoke("resuming pixel index", completed, total));
                WriteResumeIndex(_resumeIndexPath, ImageCount, pixelResumePosition);
            }

            // Anything past pixelResumePosition is a dangling, never-flushed page and gets truncated away.
            _pixelStream.SetLength(pixelResumePosition);
            _pixelStream.Position = pixelResumePosition;

            // Migrates an existing tag_rows.jsonl automatically on first open (see
            // TagRowStore.OpenForWriting), and drops any row at/past ImageCount the same
            // way the pixel truncate above does — SQLite's own transaction log already
            // makes a dangling, never-committed page impossible here (unlike the old
            // JSONL format, which needed its own manual dangling-line detection), so
            // there's nothing else to do.
            _tagRowStore = TagRowStore.OpenForWriting(directory, ImageCount,
                onMigrateProgress: onResumeProgress);
        }
        else
        {
            _pixelStream = File.Create(pixelPath);
            _pixelWriter = new BinaryWriter(_pixelStream);
            _pixelWriter.Write(PreprocessedDatasetCache.FormatMagic);
            _pixelWriter.Write(0); // image count placeholder, patched by Flush/Dispose
            _pixelWriter.Write(inputSize);

            _tagRowStore = TagRowStore.CreateFresh(directory);
            ImageCount = 0;
        }
    }

    /// <summary>Appends one resized (not yet padded/normalized) image (see Core.Encoding.ImagePreprocessing) and its tag row indices.</summary>
    public void Append(EncodedImage image, IReadOnlyList<int> tagRows)
    {
        var expectedLength = PreprocessedDatasetCache.ContentByteLength(image.Content.Width, image.Content.Height);
        if (image.Pixels.Length != expectedLength)
            throw new ArgumentException($"Expected {expectedLength} bytes for a {image.Content.Width}x{image.Content.Height} content region, got {image.Pixels.Length}.");

        _pixelWriter.Write(image.Content.X);
        _pixelWriter.Write(image.Content.Y);
        _pixelWriter.Write(image.Content.Width);
        _pixelWriter.Write(image.Content.Height);
        _pixelWriter.Write(image.Pixels);
        _tagRowStore.Append(ImageCount, tagRows);
        ImageCount++;
    }

    /// <summary>
    /// Reads the tag-row indices durably committed as of the last <see cref="Flush"/>
    /// (or resume), keyed by row index — for a caller that needs to know what tags an
    /// already-written image currently has (e.g. to merge in tags a later duplicate of
    /// the same image brings from a different source) without maintaining a second
    /// copy of that state itself.
    /// </summary>
    public IReadOnlyList<IReadOnlyList<int>> ReadCommittedTagRows(Action<int, int>? onProgress = null) =>
        _tagRowStore.ReadAll(ImageCount, onProgress);

    /// <summary>
    /// Overwrites the tag-row indices for specific already-committed rows — the only
    /// way an already-written row's labels change after the fact, e.g. merging in tags
    /// a duplicate of the same image (crawled from a different site) carries that the
    /// first copy didn't. <paramref name="tagRowsByIndex"/> gives each row's full,
    /// already-merged tag set, not just the delta. Every index must already be durably
    /// committed (&lt; <see cref="ImageCount"/> as of the last <see cref="Flush"/>) —
    /// call this right after <see cref="Flush"/>, never against a row appended earlier
    /// in the same not-yet-flushed page (see <see cref="TagRowStore.MergeRows"/> for why
    /// that ordering matters).
    /// </summary>
    public void MergeTagRows(IReadOnlyDictionary<int, IReadOnlyList<int>> tagRowsByIndex) =>
        _tagRowStore.MergeRows(tagRowsByIndex);

    /// <summary>
    /// Durably persists everything appended so far (patches the header count and
    /// flushes both files) so a crash loses at most the images appended since the
    /// last call, and a subsequent <see cref="OpenOrCreate"/> resumes right here.
    /// </summary>
    public void Flush()
    {
        // Tag rows commit BEFORE the pixel header is patched: a crash between the two
        // just leaves the pixel header (the source of truth ImageCount is read from on
        // the next open) undercounting what TagRowStore actually has committed — safe,
        // since every reader bounds its own query by ImageCount regardless. The reverse
        // order would risk ImageCount claiming more confirmed images than TagRowStore
        // actually has rows for, leaving a real gap somewhere in 0..ImageCount.
        _tagRowStore.Flush();

        _pixelWriter.Flush();
        var pixelEndOffset = _pixelStream.Position;
        _pixelStream.Position = sizeof(int); // past the magic number
        _pixelWriter.Write(ImageCount);
        _pixelStream.Position = pixelEndOffset;
        _pixelWriter.Flush();

        // pixelEndOffset is exactly the confirmed end-of-data offset for the ImageCount
        // just patched above — keep the resume index in step so the NEXT OpenOrCreate
        // (this process resuming after a crash, or a later one) can skip straight to it
        // instead of re-walking every pixel record. A crash between this write and the
        // pixel header patch above just means the index is one Flush stale next time
        // it's read — TryReadResumeIndex's ImageCount check catches that and falls back
        // to the walk, so this never needs to be perfectly atomic with the patch above.
        WriteResumeIndex(_resumeIndexPath, ImageCount, pixelEndOffset);
    }

    /// <summary>Overwrites <see cref="PreprocessedDatasetCache.ResumeIndexFileName"/> with the current answer — tiny (16 bytes), cheap to rewrite in full every call.</summary>
    private static void WriteResumeIndex(string path, int imageCount, long pixelDataEndOffset)
    {
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(stream);
        writer.Write(PreprocessedDatasetCache.ResumeIndexMagic);
        writer.Write(imageCount);
        writer.Write(pixelDataEndOffset);
    }

    /// <summary>
    /// True (with <paramref name="pixelDataEndOffset"/> populated) only if the resume
    /// index exists, isn't corrupt, and was written for exactly
    /// <paramref name="expectedImageCount"/> — a stale index (written for a smaller
    /// ImageCount, e.g. from before a subsequent crash added more confirmed images
    /// without a matching Flush reaching this file) is deliberately rejected rather
    /// than trusted, since silently resuming from the wrong offset would corrupt the
    /// pixel file. Any read failure (missing file, truncated, wrong magic) is treated
    /// the same as "no index yet" rather than thrown — this is purely an optimization
    /// the caller can always safely fall back from.
    /// </summary>
    private static bool TryReadResumeIndex(string path, int expectedImageCount, out long pixelDataEndOffset)
    {
        pixelDataEndOffset = 0;
        if (!File.Exists(path))
            return false;

        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var reader = new BinaryReader(stream);
            if (reader.ReadInt32() != PreprocessedDatasetCache.ResumeIndexMagic)
                return false;
            if (reader.ReadInt32() != expectedImageCount)
                return false;

            pixelDataEndOffset = reader.ReadInt64();
            return true;
        }
        catch (IOException)
        {
            // Covers EndOfStreamException (a truncated/corrupt index file) too — it
            // derives from IOException.
            return false;
        }
    }

    public void Dispose()
    {
        Flush();
        _pixelWriter.Dispose();
        _tagRowStore.Dispose();
    }
}

public sealed class PreprocessedDatasetCacheReader : IDisposable
{
    private readonly FileStream _pixelStream;
    private readonly long[] _offsets;
    private readonly TagRowStore _tagRowStore;

    public int ImageCount { get; }
    public int InputSize { get; }
    public IReadOnlyList<IReadOnlyList<int>> ImageTagRows { get; }

    /// <summary><paramref name="onProgress"/> (subPhase label, completed, total), if given, reports on the same two O(ImageCount) steps <see cref="PreprocessedDatasetCacheWriter.OpenOrCreate"/> does: a one-time <c>tag_rows.jsonl</c> migration (only if this cache predates <see cref="TagRowStore"/>) and loading every tag row into <see cref="ImageTagRows"/>.</summary>
    public PreprocessedDatasetCacheReader(string directory, Action<string, int, int>? onProgress = null)
    {
        _pixelStream = File.OpenRead(Path.Combine(directory, PreprocessedDatasetCache.PixelsFileName));
        using (var reader = new BinaryReader(_pixelStream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            var magic = reader.ReadInt32();
            if (magic != PreprocessedDatasetCache.FormatMagic)
                throw new InvalidDataException(
                    $"Cache at '{directory}' was built with an older cache format. Delete it and rebuild.");

            ImageCount = reader.ReadInt32();
            InputSize = reader.ReadInt32();
        }

        // Records are variable-length (content-only, no padding stored), so random
        // access needs an offset table. Building it costs one sequential pass over just
        // the box headers (16 bytes each) — never the pixel payloads — so it stays cheap
        // even at a multi-million-image corpus.
        _offsets = PreprocessedImageIndex.BuildOffsets(_pixelStream, ImageCount);

        // Migrates an existing tag_rows.jsonl automatically on first open, same as the
        // writer — training can open a cache directly that no writer in this process
        // has ever touched since TagRowStore existed.
        _tagRowStore = TagRowStore.OpenForReading(directory, ImageCount, onMigrateProgress: onProgress);
        ImageTagRows = _tagRowStore.ReadAll(ImageCount,
            onProgress: (completed, total) => onProgress?.Invoke("loading tag rows", completed, total));
    }

    /// <summary>Reads one image's letterbox content region off disk and reconstructs the full padded, normalized pixel tensor a model consumes — the cache is never fully loaded into memory.</summary>
    public PreprocessedImage ReadImage(int index)
    {
        _pixelStream.Position = _offsets[index];

        var box = new byte[PreprocessedDatasetCache.BoxBytes];
        var boxRead = _pixelStream.Read(box, 0, box.Length);
        if (boxRead != box.Length)
            throw new EndOfStreamException($"Expected a {box.Length}-byte box header for image {index}, got {boxRead}.");

        var content = new LetterboxBox(
            BitConverter.ToInt32(box, 0),
            BitConverter.ToInt32(box, 4),
            BitConverter.ToInt32(box, 8),
            BitConverter.ToInt32(box, 12));

        var pixelBytes = new byte[PreprocessedDatasetCache.ContentByteLength(content.Width, content.Height)];
        var pixelsRead = _pixelStream.Read(pixelBytes, 0, pixelBytes.Length);
        if (pixelsRead != pixelBytes.Length)
            throw new EndOfStreamException($"Expected {pixelBytes.Length} pixel bytes for image {index}, got {pixelsRead}.");

        return ImagePreprocessing.Reconstruct(new EncodedImage(pixelBytes, content), InputSize);
    }

    public void Dispose()
    {
        _pixelStream.Dispose();
        _tagRowStore.Dispose();
    }
}
