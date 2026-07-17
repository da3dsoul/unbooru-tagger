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
    internal const string LabelsFileName = "tag_rows.json";
    internal const int HeaderBytes = sizeof(int) * 2;
}

public sealed class PreprocessedDatasetCacheWriter : IDisposable
{
    private readonly FileStream _pixelStream;
    private readonly BinaryWriter _pixelWriter;
    private readonly List<int[]> _tagRows = [];
    private readonly int _inputSize;
    private readonly string _directory;

    public PreprocessedDatasetCacheWriter(string directory, int inputSize)
    {
        _directory = directory;
        _inputSize = inputSize;
        Directory.CreateDirectory(directory);

        _pixelStream = File.Create(Path.Combine(directory, PreprocessedDatasetCache.PixelsFileName));
        _pixelWriter = new BinaryWriter(_pixelStream);
        _pixelWriter.Write(0); // image count placeholder, patched in Dispose once known
        _pixelWriter.Write(inputSize);
    }

    /// <summary>Appends one already-normalized image (see Core.Encoding.ImagePreprocessing) and its tag row indices.</summary>
    public void Append(float[] normalizedPixels, IReadOnlyList<int> tagRows)
    {
        var expectedLength = 3 * _inputSize * _inputSize;
        if (normalizedPixels.Length != expectedLength)
            throw new ArgumentException($"Expected {expectedLength} floats for a {_inputSize}x{_inputSize} image, got {normalizedPixels.Length}.");

        foreach (var value in normalizedPixels)
            _pixelWriter.Write(value);
        _tagRows.Add(tagRows.ToArray());
    }

    public void Dispose()
    {
        _pixelStream.Position = 0;
        _pixelWriter.Write(_tagRows.Count);
        _pixelWriter.Flush();
        _pixelWriter.Dispose();

        File.WriteAllText(Path.Combine(_directory, PreprocessedDatasetCache.LabelsFileName), JsonSerializer.Serialize(_tagRows));
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

        var json = File.ReadAllText(Path.Combine(directory, PreprocessedDatasetCache.LabelsFileName));
        ImageTagRows = JsonSerializer.Deserialize<List<int[]>>(json)
                      ?? throw new InvalidDataException($"'{directory}' does not contain valid tag-row labels.");
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
