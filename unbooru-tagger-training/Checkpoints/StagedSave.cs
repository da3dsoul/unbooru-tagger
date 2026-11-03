namespace UnbooruTagger.Training.Checkpoints;

/// <summary>
/// Writes a checkpoint's files into a staging directory beside it, and moves them over the old ones only once every file is
/// written, training_progress.json last. Most of a save is copying the weights and optimizer state off the device: on the B70
/// the copy engine has faulted mid-run (xe "Engine memory CAT error, class=bcs"; 2026-10-24 to 11-01), and a fault while the
/// files were written in place would have left new weights beside an old optimizer or a half-written file. Now a save that
/// dies leaves the previous checkpoint whole and a staging directory the next save discards.
/// </summary>
public static class StagedSave
{
    public const string DirectoryName = ".saving";

    public static void Run(string checkpointDir, Action<string> write)
    {
        var staging = Path.Combine(checkpointDir, DirectoryName);
        if (Directory.Exists(staging))
            Directory.Delete(staging, recursive: true);
        Directory.CreateDirectory(staging);
        write(staging);
        // Each move is a rename within one directory, so a file is always either the old one or the new one.
        foreach (var file in Directory.GetFiles(staging).OrderBy(f => Path.GetFileName(f) == TrainingState.ProgressFileName))
            File.Move(file, Path.Combine(checkpointDir, Path.GetFileName(file)), overwrite: true);
        Directory.Delete(staging, recursive: true);
    }
}
