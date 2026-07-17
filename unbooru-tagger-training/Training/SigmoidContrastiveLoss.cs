using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// SigLIP-style sigmoid loss: every (image, tag) pair in the batch is an independent
/// binary match/no-match decision, not a softmax over the batch (CLAUDE.md's training
/// objective — confidence for a pair is sigmoid(dot(image, tag))).
/// </summary>
public static class SigmoidContrastiveLoss
{
    /// <param name="imageEmbeddings">[batchImages, dim]</param>
    /// <param name="tagEmbeddings">[batchTags, dim]</param>
    /// <param name="labels">[batchImages, batchTags]: +1 where the image is tagged with that tag, -1 otherwise.</param>
    public static Tensor Compute(Tensor imageEmbeddings, Tensor tagEmbeddings, Tensor labels)
    {
        var logits = imageEmbeddings.matmul(tagEmbeddings.t());
        // -log(sigmoid(labels * logits)) == softplus(-labels * logits): the standard
        // binary-cross-entropy-with-logits form of the sigmoid loss.
        return nn.functional.softplus(-labels * logits).mean();
    }
}
