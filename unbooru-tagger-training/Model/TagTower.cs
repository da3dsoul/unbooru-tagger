using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace UnbooruTagger.Training.Model;

/// <summary>
/// The tag embedding table. Splits the table into a frozen block plus a trainable
/// slice so the "add a tag" pipeline can fine-tune exactly one row with the image
/// encoder and every other row frozen (CLAUDE.md's tag-growth section), while a full
/// training pass simply makes the trainable slice the whole table.
/// </summary>
public sealed class TagTower : Module<Tensor, Tensor>
{
    private readonly Tensor _frozenRows;
    private readonly Parameter _trainableRows;
    private readonly long _trainableStart;

    public int EmbeddingDim { get; }
    public long VocabularySize => _frozenRows.shape[0];

    private TagTower(Tensor frozenRows, Parameter trainableRows, long trainableStart, int embeddingDim)
        : base(nameof(TagTower))
    {
        _frozenRows = frozenRows;
        _trainableRows = trainableRows;
        _trainableStart = trainableStart;
        EmbeddingDim = embeddingDim;

        RegisterComponents();
    }

    /// <summary>Every row trainable — the full/periodic fine-tune pass.</summary>
    public static TagTower CreateFullyTrainable(float[][] initialRows, int embeddingDim, Device? device = null)
    {
        var weights = ToTensor(initialRows, embeddingDim, device);
        return new TagTower(weights.detach(), new Parameter(weights.clone()), 0, embeddingDim);
    }

    /// <summary>
    /// Only <paramref name="trainableRowIndex"/> is trainable — the "add a tag" pipeline.
    /// <paramref name="initialRows"/>[trainableRowIndex] should already hold the
    /// text-embedding warm-start prior for the new tag.
    /// </summary>
    public static TagTower CreateWithSingleTrainableRow(float[][] initialRows, int trainableRowIndex, int embeddingDim, Device? device = null)
    {
        var weights = ToTensor(initialRows, embeddingDim, device);
        var trainableRow = weights.narrow(0, trainableRowIndex, 1).clone();
        return new TagTower(weights.detach(), new Parameter(trainableRow), trainableRowIndex, embeddingDim);
    }

    public override Tensor forward(Tensor tagIndices)
    {
        var combined = BuildCombinedTable();
        return combined.index_select(0, tagIndices);
    }

    /// <summary>Reads the current (possibly fine-tuned) table back out, e.g. before persisting to a <c>TagEmbeddingStore</c>.</summary>
    public float[][] ExtractRows()
    {
        using var combined = BuildCombinedTable();
        var flat = combined.detach().data<float>().ToArray();
        var rows = new float[VocabularySize][];
        for (var i = 0; i < rows.Length; i++)
        {
            rows[i] = new float[EmbeddingDim];
            Array.Copy(flat, i * EmbeddingDim, rows[i], 0, EmbeddingDim);
        }
        return rows;
    }

    private Tensor BuildCombinedTable()
    {
        var trainableCount = _trainableRows.shape[0];
        var totalRows = _frozenRows.shape[0];
        var trainableEnd = _trainableStart + trainableCount;

        var pieces = new List<Tensor>();
        if (_trainableStart > 0)
            pieces.Add(_frozenRows.narrow(0, 0, _trainableStart));
        pieces.Add(_trainableRows);
        if (trainableEnd < totalRows)
            pieces.Add(_frozenRows.narrow(0, trainableEnd, totalRows - trainableEnd));

        return pieces.Count == 1 ? pieces[0] : cat(pieces.ToArray(), dim: 0);
    }

    private static Tensor ToTensor(float[][] rows, int embeddingDim, Device? device)
    {
        var flat = new float[rows.Length * embeddingDim];
        for (var i = 0; i < rows.Length; i++)
            Array.Copy(rows[i], 0, flat, i * embeddingDim, embeddingDim);
        return tensor(flat, [rows.Length, embeddingDim], device: device);
    }
}
