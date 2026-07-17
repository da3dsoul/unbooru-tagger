namespace UnbooruTagger.Training.Training;

/// <summary>
/// Tracks validation performance across evaluations and decides when training should
/// stop: "train a bit, check its work, and decide when to stop" instead of always
/// running a fixed epoch/step count. Stops once <paramref name="patience"/>
/// consecutive evaluations pass without the validation loss improving by at least
/// <paramref name="minDelta"/>.
/// </summary>
public sealed class EarlyStopping(int patience = 3, double minDelta = 1e-4)
{
    private double _bestLoss = double.PositiveInfinity;
    private int _evaluationsSinceImprovement;

    public bool ShouldStop(double validationLoss)
    {
        if (validationLoss < _bestLoss - minDelta)
        {
            _bestLoss = validationLoss;
            _evaluationsSinceImprovement = 0;
            return false;
        }

        _evaluationsSinceImprovement++;
        return _evaluationsSinceImprovement >= patience;
    }
}
