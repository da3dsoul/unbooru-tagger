using System.Text.Json;
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
    internal const string LabelsFileName = "tag_rows.jsonl";

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

    /// <summary>The byte offset just past the last of <paramref name="imageCount"/> records — where a resumed writer should continue appending.</summary>
    public static long FindEndOffset(Stream stream, int imageCount)
    {
        var position = (long)PreprocessedDatasetCache.HeaderBytes;
        var box = new byte[PreprocessedDatasetCache.BoxBytes];
        for (var i = 0; i < imageCount; i++)
            position += ReadRecordLength(stream, position, box, i);

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
/// Appends to <see cref="PreprocessedDatasetCache"/>'s on-disk files. Labels are
/// stored one JSON array per line (not one big JSON array) specifically so they can
/// be appended without rewriting the whole file — needed for <see cref="OpenOrCreate"/>
/// to resume a run that was interrupted partway through a multi-million-image corpus.
/// </summary>
public sealed class PreprocessedDatasetCacheWriter : IDisposable
{
    private readonly FileStream _pixelStream;
    private readonly BinaryWriter _pixelWriter;
    private readonly string _labelPath;
    private StreamWriter _labelWriter;

    /// <summary>Images durably committed as of the last <see cref="Flush"/> (or resumed from a prior run).</summary>
    public int ImageCount { get; private set; }

    public PreprocessedDatasetCacheWriter(string directory, int inputSize)
        : this(directory, inputSize, resume: false)
    {
    }

    /// <summary>
    /// Opens an existing cache directory to continue appending where a prior run left
    /// off, or creates a fresh one if <paramref name="directory"/> is empty/missing.
    /// Any bytes/lines left over from a page that started but never finished a
    /// <see cref="Flush"/> are dropped, so resumed appends start exactly at the last
    /// confirmed image boundary.
    /// </summary>
    public static PreprocessedDatasetCacheWriter OpenOrCreate(string directory, int inputSize) =>
        new(directory, inputSize, resume: true);

    private PreprocessedDatasetCacheWriter(string directory, int inputSize, bool resume)
    {
        Directory.CreateDirectory(directory);

        var pixelPath = Path.Combine(directory, PreprocessedDatasetCache.PixelsFileName);
        var labelPath = Path.Combine(directory, PreprocessedDatasetCache.LabelsFileName);
        _labelPath = labelPath;

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
            // could — walk the ImageCount confirmed records' box headers (16 bytes each,
            // never the pixel payload) to find exactly where the last one ends. Anything
            // past that point is a dangling, never-flushed page and gets truncated away.
            var resumePosition = PreprocessedImageIndex.FindEndOffset(_pixelStream, ImageCount);
            _pixelStream.SetLength(resumePosition);
            _pixelStream.Position = resumePosition;

            var confirmedLines = File.Exists(labelPath) ? File.ReadLines(labelPath).Take(ImageCount).ToList() : [];
            File.WriteAllLines(labelPath, confirmedLines);
            _labelWriter = new StreamWriter(new FileStream(labelPath, FileMode.Append, FileAccess.Write, FileShare.Read));
        }
        else
        {
            _pixelStream = File.Create(pixelPath);
            _pixelWriter = new BinaryWriter(_pixelStream);
            _pixelWriter.Write(PreprocessedDatasetCache.FormatMagic);
            _pixelWriter.Write(0); // image count placeholder, patched by Flush/Dispose
            _pixelWriter.Write(inputSize);

            // FileShare.Read (File.Create's default is exclusive None) so
            // ReadCommittedTagRows/MergeTagRows can open their own handle on this same
            // file while this one's still held open, instead of throwing IOException.
            _labelWriter = new StreamWriter(new FileStream(labelPath, FileMode.Create, FileAccess.Write, FileShare.Read));
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
        _labelWriter.WriteLine(JsonSerializer.Serialize(tagRows));
        ImageCount++;
    }

    /// <summary>
    /// Reads the tag-row indices durably committed as of the last <see cref="Flush"/>
    /// (or resume), keyed by row index — for a caller that needs to know what tags an
    /// already-written image currently has (e.g. to merge in tags a later duplicate of
    /// the same image brings from a different source) without maintaining a second
    /// copy of that state itself.
    /// </summary>
    public IReadOnlyList<IReadOnlyList<int>> ReadCommittedTagRows() =>
        File.ReadLines(_labelPath)
            .Take(ImageCount)
            .Select(line => (IReadOnlyList<int>)(JsonSerializer.Deserialize<int[]>(line)
                             ?? throw new InvalidDataException($"'{_labelPath}' contains an invalid tag-row label line.")))
            .ToList();

    /// <summary>
    /// Overwrites the tag-row indices for specific already-committed rows — the only
    /// way an already-written row's labels change after the fact, e.g. merging in tags
    /// a duplicate of the same image (crawled from a different site) carries that the
    /// first copy didn't. <paramref name="tagRowsByIndex"/> gives each row's full,
    /// already-merged tag set, not just the delta. Every index must already be durably
    /// committed (&lt; <see cref="ImageCount"/> as of the last <see cref="Flush"/>) —
    /// call this right after <see cref="Flush"/>, never against a row appended earlier
    /// in the same not-yet-flushed page.
    ///
    /// Requires rewriting the whole labels file: JSONL lines vary in byte length, so an
    /// existing line can't be patched in place the way the fixed-width pixel file can.
    /// Only called when there's actually something to merge, not on every checkpoint —
    /// see the caller for why that keeps this affordable against a large corpus.
    /// </summary>
    public void MergeTagRows(IReadOnlyDictionary<int, IReadOnlyList<int>> tagRowsByIndex)
    {
        if (tagRowsByIndex.Count == 0)
            return;

        _labelWriter.Flush();
        _labelWriter.Dispose();

        var lines = File.ReadLines(_labelPath).Take(ImageCount).ToList();
        foreach (var (rowIndex, tagRows) in tagRowsByIndex)
            lines[rowIndex] = JsonSerializer.Serialize(tagRows);
        File.WriteAllLines(_labelPath, lines);

        _labelWriter = new StreamWriter(new FileStream(_labelPath, FileMode.Append, FileAccess.Write, FileShare.Read));
    }

    /// <summary>
    /// Durably persists everything appended so far (patches the header count and
    /// flushes both files) so a crash loses at most the images appended since the
    /// last call, and a subsequent <see cref="OpenOrCreate"/> resumes right here.
    /// </summary>
    public void Flush()
    {
        _pixelWriter.Flush();
        var position = _pixelStream.Position;
        _pixelStream.Position = sizeof(int); // past the magic number
        _pixelWriter.Write(ImageCount);
        _pixelStream.Position = position;
        _pixelWriter.Flush();

        _labelWriter.Flush();
    }

    public void Dispose()
    {
        Flush();
        _pixelWriter.Dispose();
        _labelWriter.Dispose();
    }
}

public sealed class PreprocessedDatasetCacheReader : IDisposable
{
    private readonly FileStream _pixelStream;
    private readonly long[] _offsets;

    public int ImageCount { get; }
    public int InputSize { get; }
    public IReadOnlyList<IReadOnlyList<int>> ImageTagRows { get; }

    public PreprocessedDatasetCacheReader(string directory)
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

        // Only the first ImageCount lines are confirmed committed — a crash can leave
        // dangling trailing lines from a page that started but never finished flushing.
        ImageTagRows = File.ReadLines(Path.Combine(directory, PreprocessedDatasetCache.LabelsFileName))
            .Take(ImageCount)
            .Select(line => (IReadOnlyList<int>)(JsonSerializer.Deserialize<int[]>(line)
                             ?? throw new InvalidDataException($"'{directory}' contains an invalid tag-row label line.")))
            .ToList();
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

    public void Dispose() => _pixelStream.Dispose();
}
