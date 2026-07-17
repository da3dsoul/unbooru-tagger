using UnbooruTagger.Core.Embedding;

namespace UnbooruTagger.Tests.Core;

public class TagEmbeddingStoreTests
{
    [Fact]
    public void SaveAndLoad_RoundTripsRows()
    {
        var store = TagEmbeddingStore.CreateEmpty(3);
        store.AppendRow([1f, 2f, 3f]);
        store.AppendRow([4f, 5f, 6f]);

        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.bin");
        try
        {
            store.Save(path);
            var loaded = TagEmbeddingStore.Load(path);

            Assert.Equal(3, loaded.EmbeddingDim);
            Assert.Equal(2, loaded.RowCount);
            Assert.Equal(new float[] { 1, 2, 3 }, loaded.GetRow(0).ToArray());
            Assert.Equal(new float[] { 4, 5, 6 }, loaded.GetRow(1).ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void AppendRow_RejectsWrongDimension()
    {
        var store = TagEmbeddingStore.CreateEmpty(3);
        Assert.Throws<ArgumentException>(() => store.AppendRow([1f, 2f]));
    }

    [Fact]
    public void SetRow_OverwritesExistingRow()
    {
        var store = TagEmbeddingStore.CreateEmpty(2);
        store.AppendRow([1f, 1f]);

        store.SetRow(0, [9f, 9f]);

        Assert.Equal(new float[] { 9, 9 }, store.GetRow(0).ToArray());
    }
}
