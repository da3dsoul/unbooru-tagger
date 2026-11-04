using System.Buffers.Binary;
using System.Text.Json;
using SkiaSharp;
using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Core.Dataset;

/// <summary>
/// A side store of some cache rows at a higher resolution than the cache's (2026-11-04): the cache was built with a
/// nearest-neighbour downscale to 512px, which loses small objects' lines, and re-pulling every image is too big, so
/// chosen rows are re-downloaded, downscaled properly to <see cref="LongSide"/> and kept as JPEG. Written by
/// <c>review/hires/pull.py</c> (outside git). Files in the directory:
/// <list type="bullet">
/// <item><c>store.json</c>: <c>{"LongSide": 1024}</c>.</item>
/// <item><c>images.bin</c>: the JPEG bytes, concatenated.</item>
/// <item><c>index.bin</c>: 24-byte little-endian records, one per image: int32 cache row, int64 offset into images.bin,
/// int32 length, int32 width, int32 height. A row's last record wins; a record whose bytes run past images.bin's end
/// (a write cut off, or not finished yet) is ignored until it is, so the writer appends the image before its record.</item>
/// </list>
/// Each image is stored at its letterbox content size for a <see cref="LongSide"/> canvas, so reading it at that input size
/// needs no resize.
/// </summary>
public sealed class HiResImageStore : IDisposable
{
    public const int RecordBytes = 24;

    // Replaced whole by Refresh, never changed in place, so readers on other threads always see a complete index.
    private volatile Dictionary<int, (long Offset, int Length, int Width, int Height)> _entries = new();
    private readonly Microsoft.Win32.SafeHandles.SafeFileHandle _data;

    public int LongSide { get; }
    public string Directory { get; }
    public int Count => _entries.Count;

    private HiResImageStore(string directory)
    {
        Directory = directory;
        using (var json = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "store.json"))))
            LongSide = json.RootElement.GetProperty("LongSide").GetInt32();
        _data = File.OpenHandle(Path.Combine(directory, "images.bin"), FileMode.Open, FileAccess.Read, FileShare.ReadWrite, FileOptions.RandomAccess);
        Refresh();
    }

    /// <summary>Bytes of index.bin read so far: <see cref="Refresh"/> reads on from here.</summary>
    private long _indexRead;

    /// <summary>
    /// Reads the records appended to index.bin since the last read (a pull still writing the store), returning how many rows
    /// were added. A record whose image isn't all in images.bin yet stops the read there, to be read again next time. Readers on
    /// other threads are safe (the index is swapped whole); call it from one thread at a time.
    /// </summary>
    public int Refresh()
    {
        var entries = new Dictionary<int, (long Offset, int Length, int Width, int Height)>(_entries);
        var dataLength = RandomAccess.GetLength(_data);
        using var index = new FileStream(Path.Combine(Directory, "index.bin"), FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        var available = (index.Length - _indexRead) / RecordBytes * RecordBytes;
        if (available <= 0)
            return 0;
        var bytes = new byte[available];
        index.Seek(_indexRead, SeekOrigin.Begin);
        index.ReadExactly(bytes);
        for (var at = 0; at < bytes.Length; at += RecordBytes)
        {
            var record = bytes.AsSpan(at, RecordBytes);
            var row = BinaryPrimitives.ReadInt32LittleEndian(record);
            var offset = BinaryPrimitives.ReadInt64LittleEndian(record[4..]);
            var length = BinaryPrimitives.ReadInt32LittleEndian(record[12..]);
            if (offset + length > dataLength)
                break;
            entries[row] = (offset, length, BinaryPrimitives.ReadInt32LittleEndian(record[16..]), BinaryPrimitives.ReadInt32LittleEndian(record[20..]));
            _indexRead += RecordBytes;
        }

        var added = entries.Count - _entries.Count;
        _entries = entries;
        return added;
    }

    public static HiResImageStore Open(string directory) => new(directory);

    /// <summary>Whether <paramref name="directory"/> holds a store yet (a pull writes store.json and its files when it starts one).</summary>
    public static bool Exists(string directory) =>
        File.Exists(Path.Combine(directory, "store.json")) && File.Exists(Path.Combine(directory, "images.bin")) && File.Exists(Path.Combine(directory, "index.bin"));

    public bool Contains(int row) => _entries.ContainsKey(row);

    public IEnumerable<int> Rows => _entries.Keys;

    /// <summary>The row decoded and letterboxed for an <paramref name="inputSize"/> canvas (resized only if that isn't <see cref="LongSide"/>).</summary>
    public EncodedImage Read(int row, int inputSize)
    {
        var (offset, length, _, _) = _entries[row];
        var bytes = new byte[length];
        RandomAccess.Read(_data, bytes, offset);
        using var decoded = SKBitmap.Decode(bytes) ?? throw new InvalidDataException($"Hi-res store row {row} doesn't decode.");
        using var rgba = decoded.ColorType == SKColorType.Rgba8888 ? null : decoded.Copy(SKColorType.Rgba8888);
        return ImageResize.ToEncoded(rgba ?? decoded, inputSize);
    }

    public void Dispose() => _data.Dispose();
}

/// <summary>
/// Resizing for training at an input size other than the cache's: a cached image's content scaled to a larger canvas, or a
/// hi-res store image fitted to the canvas. Unlike <see cref="ImagePreprocessing"/>'s nearest-neighbour resize (which the cache
/// and inference use, and which must not change under them), these filter: Mitchell cubic to enlarge, mipmapped linear to shrink.
/// </summary>
public static class ImageResize
{
    public static SKSamplingOptions Sampling(int from, int to) =>
        to > from ? new SKSamplingOptions(SKCubicResampler.Mitchell) : new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear);

    /// <summary>A cached image (letterboxed for a <paramref name="fromSize"/> canvas) letterboxed for a <paramref name="toSize"/> one.</summary>
    public static EncodedImage Rescale(EncodedImage image, int fromSize, int toSize)
    {
        if (fromSize == toSize)
            return image;
        using var bitmap = ToBitmap(image);
        return ToEncoded(bitmap, toSize);
    }

    /// <summary>The bitmap resized to its letterbox content size for an <paramref name="inputSize"/> canvas, as content-only RGB.</summary>
    public static EncodedImage ToEncoded(SKBitmap bitmap, int inputSize)
    {
        var content = ImagePreprocessing.ComputeLetterboxBox(bitmap.Width, bitmap.Height, inputSize);
        var resized = content.Width == bitmap.Width && content.Height == bitmap.Height
            ? null
            : bitmap.Resize(new SKImageInfo(content.Width, content.Height, SKColorType.Rgba8888, SKAlphaType.Unpremul),
                Sampling(Math.Max(bitmap.Width, bitmap.Height), Math.Max(content.Width, content.Height)))
              ?? throw new InvalidOperationException("Failed to resize image.");
        using (resized)
        {
            var source = resized ?? bitmap;
            var rgb = new byte[content.Width * content.Height * 3];
            var pixels = source.GetPixelSpan();
            for (var y = 0; y < content.Height; y++)
            for (var x = 0; x < content.Width; x++)
            {
                var from = y * source.RowBytes + x * 4;
                var to = (y * content.Width + x) * 3;
                rgb[to] = pixels[from];
                rgb[to + 1] = pixels[from + 1];
                rgb[to + 2] = pixels[from + 2];
            }

            return new EncodedImage(rgb, content);
        }
    }

    private static SKBitmap ToBitmap(EncodedImage image)
    {
        var (width, height) = (image.Content.Width, image.Content.Height);
        var bitmap = new SKBitmap(new SKImageInfo(width, height, SKColorType.Rgba8888, SKAlphaType.Unpremul));
        var pixels = bitmap.GetPixelSpan();
        for (var i = 0; i < width * height; i++)
        {
            pixels[i * 4] = image.Pixels[i * 3];
            pixels[i * 4 + 1] = image.Pixels[i * 3 + 1];
            pixels[i * 4 + 2] = image.Pixels[i * 3 + 2];
            pixels[i * 4 + 3] = 255;
        }

        return bitmap;
    }

    /// <summary>
    /// A window of cache rows at <paramref name="toSize"/>: each from <paramref name="hiRes"/> when it holds the row, else the
    /// cached image rescaled. In parallel, since decoding and resizing cost a few ms per image.
    /// </summary>
    public static EncodedImage[] Adapt(EncodedImage[] cached, IReadOnlyList<int> rows, int fromSize, int toSize, HiResImageStore? hiRes, int maxParallelism = 16) =>
        Adapt(cached, rows, fromSize, toSize, hiRes is null ? [] : [hiRes], maxParallelism);

    /// <summary>
    /// <see cref="Adapt(EncodedImage[], IReadOnlyList{int}, int, int, HiResImageStore?, int)"/> over several stores (one per long
    /// side): each row from <see cref="Pick"/>'s store, else the cached image rescaled.
    /// </summary>
    public static EncodedImage[] Adapt(EncodedImage[] cached, IReadOnlyList<int> rows, int fromSize, int toSize, IReadOnlyList<HiResImageStore> stores, int maxParallelism = 16)
    {
        var result = new EncodedImage[cached.Length];
        Parallel.For(0, cached.Length, new ParallelOptions { MaxDegreeOfParallelism = maxParallelism }, i =>
            result[i] = Pick(stores, rows[i], toSize) is { } store ? store.Read(rows[i], toSize) : Rescale(cached[i], fromSize, toSize));
        return result;
    }

    /// <summary>
    /// The store to read <paramref name="row"/> from at <paramref name="toSize"/>: of those holding it, the smallest whose long
    /// side reaches the size (least to decode, no enlarging), else the largest (enlarged less than the cache would be). Null if
    /// none holds it.
    /// </summary>
    public static HiResImageStore? Pick(IReadOnlyList<HiResImageStore> stores, int row, int toSize)
    {
        HiResImageStore? best = null;
        foreach (var store in stores)
        {
            if (!store.Contains(row))
                continue;
            if (best is null
                || (store.LongSide >= toSize && (best.LongSide < toSize || store.LongSide < best.LongSide))
                || (best.LongSide < toSize && store.LongSide > best.LongSide))
                best = store;
        }

        return best;
    }
}
