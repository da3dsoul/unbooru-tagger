using System.Buffers.Binary;
using SkiaSharp;
using UnbooruTagger.Core.Dataset;
using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Tests.Core;

public class HiResImageStoreTests
{
    private static byte[] Jpeg(int width, int height, SKColor color)
    {
        using var bitmap = new SKBitmap(width, height);
        bitmap.Erase(color);
        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(SKEncodedImageFormat.Jpeg, 95);
        return data.ToArray();
    }

    private static void Write(string directory, IEnumerable<(int Row, byte[] Jpeg, int Width, int Height)> images, int longSide, int cutLastBy = 0, bool append = false)
    {
        File.WriteAllText(Path.Combine(directory, "store.json"), $$"""{"LongSide": {{longSide}}}""");
        using var data = new FileStream(Path.Combine(directory, "images.bin"), append ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.ReadWrite);
        using var index = new FileStream(Path.Combine(directory, "index.bin"), append ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.ReadWrite);
        var record = new byte[HiResImageStore.RecordBytes];
        foreach (var (row, jpeg, width, height) in images)
        {
            BinaryPrimitives.WriteInt32LittleEndian(record, row);
            BinaryPrimitives.WriteInt64LittleEndian(record.AsSpan(4), data.Position);
            BinaryPrimitives.WriteInt32LittleEndian(record.AsSpan(12), jpeg.Length);
            BinaryPrimitives.WriteInt32LittleEndian(record.AsSpan(16), width);
            BinaryPrimitives.WriteInt32LittleEndian(record.AsSpan(20), height);
            data.Write(jpeg);
            index.Write(record);
        }

        data.SetLength(data.Length - cutLastBy);
    }

    [Fact]
    public void Read_DecodesAtTheStoreSize_AndFitsOtherSizes()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            // A 64x32 image stored for a 64px canvas: at 64 its content is 64x32, letterboxed 16px down; at 32, 32x16.
            Write(directory, [(7, Jpeg(64, 32, new SKColor(200, 40, 40)), 64, 32)], 64);
            using var store = HiResImageStore.Open(directory);

            Assert.True(store.Contains(7));
            Assert.False(store.Contains(8));
            var full = store.Read(7, 64);
            Assert.Equal(new LetterboxBox(0, 16, 64, 32), full.Content);
            Assert.Equal(64 * 32 * 3, full.Pixels.Length);
            Assert.InRange(full.Pixels[0], 190, 210);
            Assert.InRange(full.Pixels[1], 30, 50);
            var half = store.Read(7, 32);
            Assert.Equal(new LetterboxBox(0, 8, 32, 16), half.Content);
            Assert.InRange(half.Pixels[3 * 100], 190, 210);
        }
        finally
        {
            Directory.Delete(directory, true);
        }
    }

    [Fact]
    public void Open_IgnoresARecordWhoseBytesWereCutOff_AndTheLastRecordOfARowWins()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            Write(directory,
                [(1, Jpeg(8, 8, SKColors.Black), 8, 8), (1, Jpeg(8, 8, SKColors.White), 8, 8), (2, Jpeg(8, 8, SKColors.Black), 8, 8)], 8, cutLastBy: 1);
            using var store = HiResImageStore.Open(directory);

            Assert.Equal(1, store.Count);
            Assert.True(store.Read(1, 8).Pixels[0] > 240);
        }
        finally
        {
            Directory.Delete(directory, true);
        }
    }

    [Fact]
    public void Adapt_RescalesCachedRows_AndTakesStoredRowsFromTheStore()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            Write(directory, [(5, Jpeg(16, 16, SKColors.White), 16, 16)], 16);
            using var store = HiResImageStore.Open(directory);
            // Cached at an 8px canvas: a 8x4 content box, grey.
            var cached = new EncodedImage(Enumerable.Repeat((byte)100, 8 * 4 * 3).ToArray(), new LetterboxBox(0, 2, 8, 4));
            var black = new EncodedImage(new byte[8 * 8 * 3], new LetterboxBox(0, 0, 8, 8));

            var adapted = ImageResize.Adapt([cached, black], [4, 5], 8, 16, store);

            Assert.Equal(new LetterboxBox(0, 4, 16, 8), adapted[0].Content);
            Assert.All(adapted[0].Pixels, p => Assert.InRange(p, 98, 102));
            Assert.Equal(new LetterboxBox(0, 0, 16, 16), adapted[1].Content);
            Assert.True(adapted[1].Pixels[0] > 240); // from the store, not the cached black image
            Assert.Same(cached.Pixels, ImageResize.Rescale(cached, 8, 8).Pixels);
        }
        finally
        {
            Directory.Delete(directory, true);
        }
    }

    [Fact]
    public void Refresh_ReadsRecordsAppendedSince_OnceTheirImagesAreWhole()
    {
        var directory = Directory.CreateTempSubdirectory().FullName;
        try
        {
            Write(directory, [(1, Jpeg(8, 8, SKColors.Black), 8, 8)], 8);
            using var store = HiResImageStore.Open(directory);
            Assert.Equal(0, store.Refresh());

            // A pull mid-write: the record is there, its image's last byte isn't yet.
            var white = Jpeg(8, 8, SKColors.White);
            Write(directory, [(2, white, 8, 8)], 8, cutLastBy: 1, append: true);
            Assert.Equal(0, store.Refresh());
            Assert.False(store.Contains(2));

            using (var data = new FileStream(Path.Combine(directory, "images.bin"), FileMode.Append, FileAccess.Write, FileShare.ReadWrite))
                data.WriteByte(white[^1]);
            Assert.Equal(1, store.Refresh());
            Assert.True(store.Read(2, 8).Pixels[0] > 240);
            Assert.Equal(2, store.Count);
        }
        finally
        {
            Directory.Delete(directory, true);
        }
    }

    [Fact]
    public void Pick_TakesTheSmallestStoreThatReachesTheSize_ElseTheLargest()
    {
        var directories = new[] { 8, 16, 32 }.Select(_ => Directory.CreateTempSubdirectory().FullName).ToArray();
        try
        {
            Write(directories[0], [(1, Jpeg(8, 8, SKColors.Black), 8, 8), (2, Jpeg(8, 8, SKColors.Black), 8, 8)], 8);
            Write(directories[1], [(1, Jpeg(16, 16, SKColors.Black), 16, 16)], 16);
            Write(directories[2], [(1, Jpeg(32, 32, SKColors.Black), 32, 32), (3, Jpeg(32, 32, SKColors.Black), 32, 32)], 32);
            var stores = directories.Select(HiResImageStore.Open).ToArray();
            try
            {
                Assert.Equal(16, ImageResize.Pick(stores, 1, 12)!.LongSide);
                Assert.Equal(16, ImageResize.Pick(stores, 1, 16)!.LongSide);
                Assert.Equal(32, ImageResize.Pick(stores, 1, 64)!.LongSide);
                Assert.Equal(8, ImageResize.Pick(stores, 2, 16)!.LongSide);
                Assert.Equal(32, ImageResize.Pick(stores, 3, 8)!.LongSide);
                Assert.Null(ImageResize.Pick(stores, 4, 8));
            }
            finally
            {
                foreach (var store in stores)
                    store.Dispose();
            }
        }
        finally
        {
            foreach (var directory in directories)
                Directory.Delete(directory, true);
        }
    }
}
