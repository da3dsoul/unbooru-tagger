using SkiaSharp;

namespace UnbooruTagger.Core.Encoding;

/// <summary>
/// Where an image's real content landed inside a square, letterboxed canvas of some
/// size N: the sub-rectangle <c>[X, X+Width) x [Y, Y+Height)</c>, with everything
/// outside it being <see cref="ImagePreprocessing.PadColor"/> filler. Coordinates are
/// in that canvas's own pixel space (0..N), independent of the original image's size.
/// </summary>
public readonly record struct LetterboxBox(int X, int Y, int Width, int Height);

/// <summary>One decoded/resized/normalized image plus where its real content ended up (the rest is letterbox padding).</summary>
public readonly record struct PreprocessedImage(float[] Pixels, LetterboxBox Content);

/// <summary>
/// A decoded/resized image ready for on-disk storage: raw <c>uint8</c> RGB pixels
/// covering only <see cref="Content"/> (row-major, <c>Content.Width * Content.Height *
/// 3</c> bytes) — the letterbox padding around it is never materialized, since it's
/// always the same constant <see cref="ImagePreprocessing.PadColor"/> and is cheaper to
/// re-derive at load time (see <see cref="ImagePreprocessing.Reconstruct"/>) than to
/// store. Distinct from <see cref="PreprocessedImage"/>, which is the full padded,
/// normalized, float32 canvas a model actually consumes.
/// </summary>
public readonly record struct EncodedImage(byte[] Pixels, LetterboxBox Content);

/// <summary>
/// The single source of truth for image decode/resize/normalize, shared by
/// <see cref="OnnxImageEncoder"/> (inference), Training's per-epoch batch loader, and
/// the data pipeline's bulk preprocessor — they must all agree on this or a trained
/// model's normalization will silently mismatch what it sees at inference time.
/// </summary>
public static class ImagePreprocessing
{
    public static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    public static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    /// <summary>
    /// The letterbox fill color: the mean pixel CLAUDE.md's normalization already
    /// centers on, so a padded pixel normalizes to exactly 0 in every channel — as
    /// neutral/uninformative an input as this normalization can express, rather than an
    /// arbitrary color (e.g. black) that reads as a real, if unusual, pixel value.
    /// </summary>
    public static readonly SKColor PadColor = new(
        (byte)Math.Round(Mean[0] * 255),
        (byte)Math.Round(Mean[1] * 255),
        (byte)Math.Round(Mean[2] * 255));

    /// <summary>
    /// Where an <paramref name="originalWidth"/> x <paramref name="originalHeight"/>
    /// image's content lands when letterboxed (aspect-preserving, centered) into a
    /// <paramref name="canvasSize"/> x <paramref name="canvasSize"/> square — scaled up
    /// to fit inside the canvas on its longer edge, not cropped or stretched. Exposed so
    /// callers that need the same canvas at a different resolution (e.g.
    /// <c>HeatmapRefiner</c>'s refinement guide) can compute a matching box without
    /// redoing the resize itself.
    /// </summary>
    public static LetterboxBox ComputeLetterboxBox(int originalWidth, int originalHeight, int canvasSize)
    {
        var scale = Math.Min(canvasSize / (double)originalWidth, canvasSize / (double)originalHeight);
        var contentWidth = Math.Clamp((int)Math.Round(originalWidth * scale), 1, canvasSize);
        var contentHeight = Math.Clamp((int)Math.Round(originalHeight * scale), 1, canvasSize);
        var x = (canvasSize - contentWidth) / 2;
        var y = (canvasSize - contentHeight) / 2;
        return new LetterboxBox(x, y, contentWidth, contentHeight);
    }

    /// <summary>
    /// Which locations of a <paramref name="spatialHeight"/> x <paramref name="spatialWidth"/>
    /// grid over a <paramref name="canvasSize"/> x <paramref name="canvasSize"/> letterboxed
    /// canvas fall inside <paramref name="content"/> (real image data) rather than letterbox
    /// padding — a location counts as valid if its receptive field's center falls inside
    /// <paramref name="content"/>. Shared by <see cref="OnnxImageEncoder"/> (masked
    /// inference pooling) and Training's <c>SpatialMask</c> (masked training
    /// pooling/loss) so both agree on exactly the same rule for the same canvas geometry.
    /// </summary>
    /// <remarks>
    /// An extreme aspect ratio (e.g. a 100:1 banner) can letterbox content thinner than
    /// one grid cell's stride, so no cell center falls inside it — that would otherwise
    /// leave every location invalid, which NaNs a log-sum-exp pool over an all-masked
    /// row. Falls back to the single cell nearest <paramref name="content"/>'s own
    /// center instead of leaving the grid empty; localization for that image is coarse
    /// either way.
    /// </remarks>
    public static bool[,] ComputeSpatialValidity(LetterboxBox content, int canvasSize, int spatialHeight, int spatialWidth)
    {
        var strideY = canvasSize / (float)spatialHeight;
        var strideX = canvasSize / (float)spatialWidth;
        var validity = new bool[spatialHeight, spatialWidth];

        var validCells = 0;
        for (var y = 0; y < spatialHeight; y++)
        {
            var centerY = (y + 0.5f) * strideY;
            if (centerY < content.Y || centerY >= content.Y + content.Height)
                continue;

            for (var x = 0; x < spatialWidth; x++)
            {
                var centerX = (x + 0.5f) * strideX;
                if (centerX < content.X || centerX >= content.X + content.Width)
                    continue;

                validity[y, x] = true;
                validCells++;
            }
        }

        if (validCells == 0)
        {
            var nearestY = Math.Clamp((int)((content.Y + content.Height / 2f) / strideY), 0, spatialHeight - 1);
            var nearestX = Math.Clamp((int)((content.X + content.Width / 2f) / strideX), 0, spatialWidth - 1);
            validity[nearestY, nearestX] = true;
        }

        return validity;
    }

    /// <summary>
    /// Letterboxes <paramref name="original"/> into a <paramref name="size"/> x
    /// <paramref name="size"/> canvas: scaled to fit inside on its longer edge
    /// (preserving aspect ratio, unlike a plain squash-to-square resize), centered, and
    /// padded with <see cref="PadColor"/>. Shared by <see cref="Normalize"/> (the actual
    /// model input) and <c>HeatmapRefiner.BuildGuide</c> (which needs an RGB canvas in
    /// the exact same coordinate frame as the model saw, just at a different
    /// resolution) so both agree on where real content sits.
    /// </summary>
    public static SKBitmap BuildLetterboxCanvas(SKBitmap original, int size, out LetterboxBox content)
    {
        content = ComputeLetterboxBox(original.Width, original.Height, size);

        var canvas = new SKBitmap(new SKImageInfo(size, size, SKColorType.Rgba8888, SKAlphaType.Unpremul));
        using (var skCanvas = new SKCanvas(canvas))
        {
            skCanvas.Clear(PadColor);

            var scaledInfo = new SKImageInfo(content.Width, content.Height, SKColorType.Rgba8888, SKAlphaType.Unpremul);
            using var scaled = original.Resize(scaledInfo, SKSamplingOptions.Default)
                ?? throw new InvalidOperationException("Failed to resize image.");
            skCanvas.DrawBitmap(scaled, content.X, content.Y);
        }

        return canvas;
    }

    /// <summary>Decodes, letterboxes, and normalizes one image into a flat NCHW-ordered (channels-first) <c>float[3 * inputSize * inputSize]</c>.</summary>
    public static PreprocessedImage LoadAndNormalize(string imagePath, int inputSize)
    {
        using var original = SKBitmap.Decode(imagePath)
            ?? throw new InvalidDataException($"Could not decode image at '{imagePath}'.");
        return Normalize(original, inputSize);
    }

    /// <summary>Same as <see cref="LoadAndNormalize(string, int)"/> but decodes from an in-memory stream (e.g. a DB blob) instead of a file path.</summary>
    public static PreprocessedImage LoadAndNormalize(Stream imageStream, int inputSize)
    {
        using var original = SKBitmap.Decode(imageStream)
            ?? throw new InvalidDataException("Could not decode image from the given stream.");
        return Normalize(original, inputSize);
    }

    /// <summary>
    /// Decodes and resizes an image for cache storage: unlike <see cref="LoadAndNormalize(string, int)"/>,
    /// this skips building the full padded canvas entirely and normalizing — it resizes
    /// straight to <see cref="LetterboxBox.Width"/> x <see cref="LetterboxBox.Height"/>
    /// (no padding pixels materialized) and keeps raw <c>0..255</c> byte values, since
    /// <see cref="PreprocessedDatasetCache"/> re-pads and normalizes on read instead of
    /// storing that redundant, reconstructible data.
    /// </summary>
    public static EncodedImage LoadAndEncode(string imagePath, int inputSize)
    {
        using var original = SKBitmap.Decode(imagePath)
            ?? throw new InvalidDataException($"Could not decode image at '{imagePath}'.");
        return Encode(original, inputSize);
    }

    /// <summary>Same as <see cref="LoadAndEncode(string, int)"/> but decodes from an in-memory stream (e.g. a DB blob) instead of a file path.</summary>
    public static EncodedImage LoadAndEncode(Stream imageStream, int inputSize)
    {
        using var original = SKBitmap.Decode(imageStream)
            ?? throw new InvalidDataException("Could not decode image from the given stream.");
        return Encode(original, inputSize);
    }

    /// <summary>
    /// Same as <see cref="LoadAndEncode(string, int)"/> but for a caller that's already
    /// decoded the bitmap itself — e.g. because it also needs to run something else
    /// (like a perceptual hash) against the same image and decoding twice would be pure
    /// waste. Does not take ownership of <paramref name="original"/>; the caller disposes it.
    /// </summary>
    public static EncodedImage Encode(SKBitmap original, int inputSize)
    {
        var content = ComputeLetterboxBox(original.Width, original.Height, inputSize);

        var scaledInfo = new SKImageInfo(content.Width, content.Height, SKColorType.Rgba8888, SKAlphaType.Unpremul);
        using var scaled = original.Resize(scaledInfo, SKSamplingOptions.Default)
            ?? throw new InvalidOperationException("Failed to resize image.");

        var rgb = new byte[content.Width * content.Height * 3];
        var pixels = scaled.GetPixelSpan();
        var rowBytes = scaled.RowBytes;
        const int bytesPerPixel = 4;

        for (var y = 0; y < content.Height; y++)
        {
            var rowStart = y * rowBytes;
            var outRowStart = y * content.Width * 3;
            for (var x = 0; x < content.Width; x++)
            {
                var offset = rowStart + x * bytesPerPixel;
                var outOffset = outRowStart + x * 3;
                rgb[outOffset] = pixels[offset];
                rgb[outOffset + 1] = pixels[offset + 1];
                rgb[outOffset + 2] = pixels[offset + 2];
            }
        }

        return new EncodedImage(rgb, content);
    }

    /// <summary>
    /// Re-pads and normalizes a stored <see cref="EncodedImage"/> back into the full
    /// <paramref name="inputSize"/> x <paramref name="inputSize"/> tensor a model
    /// consumes. Padding is left as the default <c>0f</c> rather than computed from
    /// <see cref="PadColor"/> pixel-by-pixel: normalizing <see cref="PadColor"/> lands
    /// only approximately at 0 (byte rounding), while a bare zero-filled array is exactly
    /// the "neutral" value that color was chosen to approximate — and it's free, since
    /// <c>new float[...]</c> is already zeroed.
    /// </summary>
    public static PreprocessedImage Reconstruct(EncodedImage encoded, int inputSize)
    {
        var content = encoded.Content;
        var expectedLength = content.Width * content.Height * 3;
        if (encoded.Pixels.Length != expectedLength)
            throw new ArgumentException($"Expected {expectedLength} bytes for a {content.Width}x{content.Height} content region, got {encoded.Pixels.Length}.");

        var channelSize = inputSize * inputSize;
        var flat = new float[3 * channelSize];

        for (var y = 0; y < content.Height; y++)
        {
            var canvasY = content.Y + y;
            var rowStart = y * content.Width * 3;
            for (var x = 0; x < content.Width; x++)
            {
                var canvasX = content.X + x;
                var pixelIndex = canvasY * inputSize + canvasX;
                var offset = rowStart + x * 3;
                flat[(0 * channelSize) + pixelIndex] = (encoded.Pixels[offset] / 255f - Mean[0]) / Std[0];
                flat[(1 * channelSize) + pixelIndex] = (encoded.Pixels[offset + 1] / 255f - Mean[1]) / Std[1];
                flat[(2 * channelSize) + pixelIndex] = (encoded.Pixels[offset + 2] / 255f - Mean[2]) / Std[2];
            }
        }

        return new PreprocessedImage(flat, content);
    }

    /// <summary>
    /// Inverse of <see cref="Reconstruct"/>: recovers an <see cref="EncodedImage"/>
    /// (raw uint8, content-only) from an already-normalized, padded
    /// <see cref="PreprocessedImage"/> — the padding is simply dropped (it was never
    /// real data), and each content pixel's normalization is undone with a round instead
    /// of re-decoding the original source image. Used only by
    /// <see cref="Dataset.PreprocessedDatasetCacheMigrator"/> to shrink an old-format
    /// cache: the crawler never keeps the original downloaded bytes around, so this is
    /// the only way to recover them without re-downloading everything.
    /// </summary>
    public static EncodedImage Strip(PreprocessedImage image, int inputSize)
    {
        var content = image.Content;
        var channelSize = inputSize * inputSize;
        var rgb = new byte[content.Width * content.Height * 3];

        for (var y = 0; y < content.Height; y++)
        {
            var canvasY = content.Y + y;
            var outRowStart = y * content.Width * 3;
            for (var x = 0; x < content.Width; x++)
            {
                var canvasX = content.X + x;
                var pixelIndex = canvasY * inputSize + canvasX;
                var outOffset = outRowStart + x * 3;
                rgb[outOffset] = Denormalize(image.Pixels[(0 * channelSize) + pixelIndex], Mean[0], Std[0]);
                rgb[outOffset + 1] = Denormalize(image.Pixels[(1 * channelSize) + pixelIndex], Mean[1], Std[1]);
                rgb[outOffset + 2] = Denormalize(image.Pixels[(2 * channelSize) + pixelIndex], Mean[2], Std[2]);
            }
        }

        return new EncodedImage(rgb, content);
    }

    private static byte Denormalize(float normalized, float mean, float std) =>
        (byte)Math.Clamp(Math.Round(((normalized * std) + mean) * 255f), 0, 255);

    private static PreprocessedImage Normalize(SKBitmap original, int inputSize)
    {
        using var canvas = BuildLetterboxCanvas(original, inputSize, out var content);

        var channelSize = inputSize * inputSize;
        var flat = new float[3 * channelSize];

        // Direct pixel-buffer access instead of GetPixel(x, y) per pixel: GetPixel's
        // per-call overhead (bounds checks, color conversion) dwarfs decode+resize cost
        // for a 224x224+ image, and this is the hottest loop in the whole preprocessing
        // pipeline since it runs once per image in the entire corpus.
        var pixels = canvas.GetPixelSpan();
        var rowBytes = canvas.RowBytes;
        const int bytesPerPixel = 4;

        for (var y = 0; y < inputSize; y++)
        {
            var rowStart = y * rowBytes;
            for (var x = 0; x < inputSize; x++)
            {
                var offset = rowStart + x * bytesPerPixel;
                var pixelIndex = y * inputSize + x;
                flat[(0 * channelSize) + pixelIndex] = (pixels[offset] / 255f - Mean[0]) / Std[0];
                flat[(1 * channelSize) + pixelIndex] = (pixels[offset + 1] / 255f - Mean[1]) / Std[1];
                flat[(2 * channelSize) + pixelIndex] = (pixels[offset + 2] / 255f - Mean[2]) / Std[2];
            }
        }

        return new PreprocessedImage(flat, content);
    }
}
