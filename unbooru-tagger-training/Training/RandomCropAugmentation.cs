using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Builds an augmented view of an already-loaded, already-normalized image batch: per
/// image, crops a random sub-region and pastes it at a random offset into an otherwise
/// blank canvas of the same size, plus a random horizontal flip. Used to build the two
/// independent views <see cref="SelfSupervisedConsistencyLoss"/> compares. Deliberately
/// works in tensor space on whatever ImageBatchLoader/PreprocessedDatasetCacheReader
/// already produced instead of touching the image decode path, so it needs no changes
/// there and behaves identically for both the raw-manifest and cache training paths.
/// </summary>
public static class RandomCropAugmentation
{
    public static Tensor Apply(Tensor pixelBatch, double minScale = 0.6, Random? random = null)
    {
        random ??= Random.Shared;

        var batch = (int)pixelBatch.shape[0];
        var height = (int)pixelBatch.shape[2];
        var width = (int)pixelBatch.shape[3];

        var output = zeros_like(pixelBatch);
        for (var i = 0; i < batch; i++)
        {
            var scale = minScale + random.NextDouble() * (1 - minScale);
            var cropHeight = Math.Clamp((int)(height * scale), 1, height);
            var cropWidth = Math.Clamp((int)(width * scale), 1, width);

            var sourceTop = random.Next(height - cropHeight + 1);
            var sourceLeft = random.Next(width - cropWidth + 1);
            var destTop = random.Next(height - cropHeight + 1);
            var destLeft = random.Next(width - cropWidth + 1);

            var crop = pixelBatch.narrow(0, i, 1).narrow(2, sourceTop, cropHeight).narrow(3, sourceLeft, cropWidth);
            if (random.Next(2) == 0)
                crop = crop.flip([3]);

            output.narrow(0, i, 1).narrow(2, destTop, cropHeight).narrow(3, destLeft, cropWidth).copy_(crop);
        }

        return output;
    }
}
