namespace UnbooruTagger.Training.Model;

public static class EmbeddingInit
{
    /// <summary>A small random row for a tag with no warm-start prior available.</summary>
    public static float[] RandomRow(int embeddingDim, Random? random = null)
    {
        random ??= Random.Shared;
        var row = new float[embeddingDim];
        for (var i = 0; i < embeddingDim; i++)
            row[i] = (float)(random.NextDouble() * 0.04 - 0.02);
        return row;
    }
}
