using System.Text.Json;
using System.Text.Json.Serialization;
using UnbooruTagger.Core.Vocabulary;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// Which tags the whole-image classifier is trusted to teach localization on (crop
/// consistency): localizable tags whose held-out AP in the latest epoch evaluation is at least
/// a threshold. Run 2's first crop-consistency trial trained every tag from epoch 1 and made
/// localization worse; a classifier that still recognizes a tag from context (hair, outfit,
/// style) says "yes" on crops that don't contain it. Gated on measured AP, the loss waits
/// until a tag is learned well and the trusted set grows by itself as training improves.
/// A minimum number of held-out positives keeps rare tags' noisy AP (often 1.0 on a handful
/// of images) from qualifying them; at epoch 1 of run 2 this left about 40 tags. A tag
/// with context sources (a character who usually wears the hat) must also tell positives
/// from that context without the tag (context AP), or a classifier answering "this
/// character, so hat" would qualify on whole-image AP alone.
/// </summary>
public sealed class CropTeacherGate
{
    public bool[] Trusted { get; }
    public int TrustedCount { get; }

    private CropTeacherGate(bool[] trusted)
    {
        Trusted = trusted;
        TrustedCount = trusted.Count(t => t);
    }

    /// <summary>Fewer context negatives than this and the context-AP requirement is skipped (too few to measure).</summary>
    public const int MinContextNegatives = 10;

    public static CropTeacherGate None(int vocabularySize) => new(new bool[vocabularySize]);

    public static CropTeacherGate From(EvaluationReport report, TagVocabulary vocabulary, bool[] localizable, double minAveragePrecision, int minPositives)
    {
        var trusted = new bool[vocabulary.Records.Count];
        foreach (var tag in report.Tags)
            if (tag.AveragePrecision >= minAveragePrecision && tag.Positives >= minPositives
                && (tag.ContextNegatives < MinContextNegatives || tag.ContextAveragePrecision >= minAveragePrecision)
                && vocabulary.TryGet(tag.Tag, out var record) && localizable[record.RowIndex])
                trusted[record.RowIndex] = true;
        return new CropTeacherGate(trusted);
    }

    /// <summary>The gate from a saved evaluation_report.json, or no trusted tags if there isn't one yet.</summary>
    public static CropTeacherGate Load(string reportPath, TagVocabulary vocabulary, bool[] localizable, double minAveragePrecision, int minPositives)
    {
        if (!File.Exists(reportPath))
            return None(vocabulary.Records.Count);
        var report = JsonSerializer.Deserialize<EvaluationReport>(File.ReadAllText(reportPath),
            new JsonSerializerOptions { NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals });
        return report is null ? None(vocabulary.Records.Count) : From(report, vocabulary, localizable, minAveragePrecision, minPositives);
    }
}
