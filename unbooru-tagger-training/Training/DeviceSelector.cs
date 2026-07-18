using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// TorchSharp modules place themselves on a device via a `device:` parameter passed to
/// each layer factory at construction time — there's no `Module.to(device)` in this
/// version to move an already-built module afterward. Picks CUDA when available.
/// </summary>
public static class DeviceSelector
{
    public static Device Best()
    {
        if (cuda_is_available())
        {
            Console.WriteLine("CUDA available — training on GPU.");
            return CUDA;
        }

        Console.WriteLine("CUDA not available — training on CPU.");
        return CPU;
    }
}
