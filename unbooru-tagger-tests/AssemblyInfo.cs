using Xunit;

// TorchSharp's RNG (torch.manual_seed / rand / randn) is process-global native state, not
// per-thread. xUnit parallelizes test classes across threads by default, so any test that
// seeds and expects reproducible random tensors (e.g. ImageTowerOnnxExportTests) can race
// against another TorchSharp-using test class and non-deterministically fail.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
