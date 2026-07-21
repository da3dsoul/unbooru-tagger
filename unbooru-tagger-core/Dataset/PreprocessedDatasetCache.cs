using System.Text.Json;

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
    internal const int HeaderBytes = sizeof(int) * 2;
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
    private readonly int _inputSize;

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
        _inputSize = inputSize;
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
                ImageCount = headerReader.ReadInt32();
                var storedInputSize = headerReader.ReadInt32();
                if (storedInputSize != inputSize)
                    throw new InvalidDataException($"Cache at '{directory}' was built with input size {storedInputSize}, but {inputSize} was requested.");
            }

            var imageBytes = 3L * inputSize * inputSize * sizeof(float);
            _pixelStream.SetLength(PreprocessedDatasetCache.HeaderBytes + ImageCount * imageBytes);
            _pixelStream.Position = _pixelStream.Length;

            var confirmedLines = File.Exists(labelPath) ? File.ReadLines(labelPath).Take(ImageCount).ToList() : [];
            File.WriteAllLines(labelPath, confirmedLines);
            _labelWriter = new StreamWriter(new FileStream(labelPath, FileMode.Append, FileAccess.Write, FileShare.Read));
        }
        else
        {
            _pixelStream = File.Create(pixelPath);
            _pixelWriter = new BinaryWriter(_pixelStream);
            _pixelWriter.Write(0); // image count placeholder, patched by Flush/Dispose
            _pixelWriter.Write(inputSize);

            // FileShare.Read (File.Create's default is exclusive None) so
            // ReadCommittedTagRows/MergeTagRows can open their own handle on this same
            // file while this one's still held open, instead of throwing IOException.
            _labelWriter = new StreamWriter(new FileStream(labelPath, FileMode.Create, FileAccess.Write, FileShare.Read));
            ImageCount = 0;
        }
    }

    /// <summary>Appends one already-normalized image (see Core.Encoding.ImagePreprocessing) and its tag row indices.</summary>
    public void Append(float[] normalizedPixels, IReadOnlyList<int> tagRows)
    {
        var expectedLength = 3 * _inputSize * _inputSize;
        if (normalizedPixels.Length != expectedLength)
            throw new ArgumentException($"Expected {expectedLength} floats for a {_inputSize}x{_inputSize} image, got {normalizedPixels.Length}.");

        foreach (var value in normalizedPixels)
            _pixelWriter.Write(value);
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
        _pixelStream.Position = 0;
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
    private readonly int _imageFloats;
    private readonly int _imageBytes;

    public int ImageCount { get; }
    public int InputSize { get; }
    public IReadOnlyList<IReadOnlyList<int>> ImageTagRows { get; }

    public PreprocessedDatasetCacheReader(string directory)
    {
        _pixelStream = File.OpenRead(Path.Combine(directory, PreprocessedDatasetCache.PixelsFileName));
        using (var reader = new BinaryReader(_pixelStream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            ImageCount = reader.ReadInt32();
            InputSize = reader.ReadInt32();
        }

        _imageFloats = 3 * InputSize * InputSize;
        _imageBytes = _imageFloats * sizeof(float);

        // Only the first ImageCount lines are confirmed committed — a crash can leave
        // dangling trailing lines from a page that started but never finished flushing.
        ImageTagRows = File.ReadLines(Path.Combine(directory, PreprocessedDatasetCache.LabelsFileName))
            .Take(ImageCount)
            .Select(line => (IReadOnlyList<int>)(JsonSerializer.Deserialize<int[]>(line)
                             ?? throw new InvalidDataException($"'{directory}' contains an invalid tag-row label line.")))
            .ToList();
    }

    /// <summary>Reads one image's normalized pixel tensor directly off disk — the cache is never fully loaded into memory.</summary>
    public float[] ReadImage(int index)
    {
        _pixelStream.Position = PreprocessedDatasetCache.HeaderBytes + ((long)index * _imageBytes);

        var buffer = new byte[_imageBytes];
        var read = _pixelStream.Read(buffer, 0, buffer.Length);
        if (read != buffer.Length)
            throw new EndOfStreamException($"Expected {buffer.Length} bytes for image {index}, got {read}.");

        var floats = new float[_imageFloats];
        Buffer.BlockCopy(buffer, 0, floats, 0, buffer.Length);
        return floats;
    }

    public void Dispose() => _pixelStream.Dispose();
}
