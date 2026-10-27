using System.Text;
using SkiaSharp;

namespace UnbooruTagger.Training.Commands;

/// <summary>
/// Encodes a bitmap as a Sixel image (DEC graphics, supported by Konsole, xterm, WezTerm, foot,
/// Windows Terminal and tmux 3.4+ built with sixel). Sixel allows 256 colors per image, so the
/// palette is the image's 256 most common colors (counted at 5 bits per channel) and every pixel
/// takes the nearest of them: charts are large flat areas plus a handful of line colors and
/// their antialiased edges, which that keeps nearly exact.
/// </summary>
public static class Sixel
{
    public static string Encode(SKBitmap bitmap)
    {
        var (width, height) = (bitmap.Width, bitmap.Height);
        var pixels = bitmap.Pixels;
        var (palette, indices) = Quantize(pixels);

        var output = new StringBuilder(width * height / 4);
        output.Append("\u001bP0;1;0q"); // pixel aspect 1:1; unpainted pixels keep the background
        output.Append($"\"1;1;{width};{height}");
        for (var c = 0; c < palette.Length; c++)
            output.Append($"#{c};2;{palette[c].Red * 100 / 255};{palette[c].Green * 100 / 255};{palette[c].Blue * 100 / 255}");

        var rows = new Dictionary<int, byte[]>();
        for (var band = 0; band < height; band += 6)
        {
            rows.Clear();
            for (var dy = 0; dy < 6 && band + dy < height; dy++)
            {
                var offset = (band + dy) * width;
                for (var x = 0; x < width; x++)
                {
                    var color = indices[offset + x];
                    if (!rows.TryGetValue(color, out var row))
                        rows[color] = row = new byte[width];
                    row[x] |= (byte)(1 << dy);
                }
            }

            var first = true;
            foreach (var (color, row) in rows)
            {
                if (!first)
                    output.Append('$'); // back to the start of this band, for the next color
                first = false;
                output.Append('#').Append(color);
                AppendRun(output, row);
            }
            output.Append('-'); // next band
        }
        output.Append("\u001b\\");
        return output.ToString();
    }

    /// <summary>One color's six-pixel columns across the band, run-length encoded (<c>!n</c>).</summary>
    private static void AppendRun(StringBuilder output, byte[] row)
    {
        // Trailing empty columns needn't be sent.
        var end = row.Length;
        while (end > 0 && row[end - 1] == 0)
            end--;
        for (var x = 0; x < end;)
        {
            var value = row[x];
            var run = 1;
            while (x + run < end && row[x + run] == value)
                run++;
            var ch = (char)('?' + value);
            if (run > 3)
                output.Append('!').Append(run).Append(ch);
            else
                output.Append(ch, run);
            x += run;
        }
    }

    /// <summary>The palette, and each pixel's index into it.</summary>
    public static (SKColor[] Palette, byte[] Indices) Quantize(SKColor[] pixels)
    {
        static int Key(SKColor c) => (c.Red >> 3) << 10 | (c.Green >> 3) << 5 | c.Blue >> 3;
        var counts = new int[1 << 15];
        var sums = new long[1 << 15, 3];
        foreach (var p in pixels)
        {
            var k = Key(p);
            counts[k]++;
            (sums[k, 0], sums[k, 1], sums[k, 2]) = (sums[k, 0] + p.Red, sums[k, 1] + p.Green, sums[k, 2] + p.Blue);
        }
        // Each bin is represented by the mean of the pixels in it.
        var palette = Enumerable.Range(0, counts.Length).Where(k => counts[k] > 0).OrderByDescending(k => counts[k]).Take(256)
            .Select(k => new SKColor((byte)(sums[k, 0] / counts[k]), (byte)(sums[k, 1] / counts[k]), (byte)(sums[k, 2] / counts[k])))
            .ToArray();

        var nearest = new short[1 << 15];
        Array.Fill(nearest, (short)-1);
        var indices = new byte[pixels.Length];
        for (var i = 0; i < pixels.Length; i++)
        {
            var k = Key(pixels[i]);
            if (nearest[k] < 0)
            {
                var (best, bestDistance) = (0, int.MaxValue);
                for (var c = 0; c < palette.Length; c++)
                {
                    int dr = palette[c].Red - pixels[i].Red, dg = palette[c].Green - pixels[i].Green, db = palette[c].Blue - pixels[i].Blue;
                    var distance = 2 * dr * dr + 4 * dg * dg + 3 * db * db;
                    if (distance < bestDistance)
                        (best, bestDistance) = (c, distance);
                }
                nearest[k] = (short)best;
            }
            indices[i] = (byte)nearest[k];
        }
        return (palette, indices);
    }
}
