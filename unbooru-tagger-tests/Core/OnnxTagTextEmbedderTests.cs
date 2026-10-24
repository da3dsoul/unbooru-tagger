using Google.Protobuf;
using Onnx;
using UnbooruTagger.Core.Encoding;

namespace UnbooruTagger.Tests.Core;

public class OnnxTagTextEmbedderTests
{
    private const int VocabSize = 5;
    private const int EmbeddingDim = 4;

    [Fact]
    public void Embed_ReturnsTheMeanPooledRowsForTheGivenTokenIds()
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.onnx");
        try
        {
            // Row i of the lookup table is filled with the constant value i, so
            // mean-pooling token ids [1, 3] should give a vector of all 2.0 -- the
            // average of 1 and 3 -- independent of the (stubbed) tokenizer's real logic.
            BuildTinyEmbeddingOnnx(path, VocabSize, EmbeddingDim);
            using var embedder = new OnnxTagTextEmbedder(path, EmbeddingDim, _ => [1, 3]);

            var embedding = embedder.Embed("whatever -- tokenize is stubbed above");

            Assert.All(embedding, value => Assert.Equal(2.0f, value, 4));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Embed_ThrowsWhenTheGraphsOutputDimDoesNotMatchTheDeclaredEmbeddingDim()
    {
        var path = Path.Combine(Path.GetTempPath(), $"{Guid.NewGuid()}.onnx");
        try
        {
            BuildTinyEmbeddingOnnx(path, VocabSize, EmbeddingDim);
            using var embedder = new OnnxTagTextEmbedder(path, EmbeddingDim + 1, _ => [0]);

            Assert.Throws<InvalidDataException>(() => embedder.Embed("tag"));
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>A tiny ONNX graph: a [vocabSize, embeddingDim] lookup table where row i is filled with the constant value i, gathered by "input_ids" and mean-pooled over the sequence axis into "embedding" -- exercises the real Gather/ReduceMean shapes OnnxTagTextEmbedder.Embed sends a token-id tensor through.</summary>
    private static void BuildTinyEmbeddingOnnx(string path, int vocabSize, int embeddingDim)
    {
        var table = new float[vocabSize * embeddingDim];
        for (var row = 0; row < vocabSize; row++)
        for (var col = 0; col < embeddingDim; col++)
            table[row * embeddingDim + col] = row;

        var graph = new GraphProto { Name = "tiny_text_embedder" };

        graph.Input.Add(new ValueInfoProto
        {
            Name = "input_ids",
            Type = new TypeProto { TensorType = new TypeProto.Types.Tensor { ElemType = (int)TensorProto.Types.DataType.Int64 } }
        });

        const string tableName = "embedding_table";
        graph.Initializer.Add(new TensorProto
        {
            Name = tableName,
            DataType = (int)TensorProto.Types.DataType.Float,
            Dims = { vocabSize, embeddingDim },
            RawData = ByteString.CopyFrom(ToBytes(table))
        });

        graph.Node.Add(new NodeProto { OpType = "Gather", Name = "Gather_0", Input = { tableName, "input_ids" }, Output = { "gathered" } });

        var reduceNode = new NodeProto { OpType = "ReduceMean", Name = "ReduceMean_0", Input = { "gathered" }, Output = { "embedding" } };
        reduceNode.Attribute.Add(new AttributeProto { Name = "axes", Type = AttributeProto.Types.AttributeType.Ints, Ints = { 1 } });
        reduceNode.Attribute.Add(new AttributeProto { Name = "keepdims", Type = AttributeProto.Types.AttributeType.Int, I = 0 });
        graph.Node.Add(reduceNode);

        graph.Output.Add(new ValueInfoProto
        {
            Name = "embedding",
            Type = new TypeProto { TensorType = new TypeProto.Types.Tensor { ElemType = (int)TensorProto.Types.DataType.Float } }
        });

        var model = new ModelProto { IrVersion = 9, Graph = graph };
        model.OpsetImport.Add(new OperatorSetIdProto { Domain = "", Version = 13 });

        using var stream = File.Create(path);
        model.WriteTo(stream);
    }

    private static byte[] ToBytes(float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
