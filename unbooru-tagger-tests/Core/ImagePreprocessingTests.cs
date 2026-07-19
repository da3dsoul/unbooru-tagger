using SkiaSharp;
using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Tests.Core;

public class ImagePreprocessingTests
{
    [Fact]
    public void LoadAndNormalize_ProducesExpectedChannelValuesAndNchwLayout()
    {
        const int inputSize = 4;
        using var bitmap = new SKBitmap(new SKImageInfo(8, 8, SKColorType.Rgba8888, SKAlphaType.Unpremul));
        using (var canvas = new SKCanvas(bitmap))
            canvas.Clear(new SKColor(200, 100, 50, 255));

        using var image = SKImage.FromBitmap(bitmap);
        using var encoded = image.Encode(SKEncodedImageFormat.Png, 100);
        using var stream = new MemoryStream();
        encoded.SaveTo(stream);
        stream.Position = 0;

        var pixels = ImagePreprocessing.LoadAndNormalize(stream, inputSize);

        var expectedR = (200 / 255f - ImagePreprocessing.Mean[0]) / ImagePreprocessing.Std[0];
        var expectedG = (100 / 255f - ImagePreprocessing.Mean[1]) / ImagePreprocessing.Std[1];
        var expectedB = (50 / 255f - ImagePreprocessing.Mean[2]) / ImagePreprocessing.Std[2];

        var channelSize = inputSize * inputSize;
        Assert.Equal(3 * channelSize, pixels.Length);
        for (var i = 0; i < channelSize; i++)
        {
            Assert.Equal(expectedR, pixels[i], 3);
            Assert.Equal(expectedG, pixels[channelSize + i], 3);
            Assert.Equal(expectedB, pixels[(2 * channelSize) + i], 3);
        }
    }
}
