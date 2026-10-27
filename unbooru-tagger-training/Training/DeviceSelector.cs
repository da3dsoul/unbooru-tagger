using TorchSharp;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Training;

/// <summary>
/// TorchSharp modules place themselves on a device via a `device:` parameter passed to
/// each layer factory at construction time — there's no `Module.to(device)` in this
/// version to move an already-built module afterward. Picks CUDA when available, then an
/// Intel GPU (XPU, only with an XPU libtorch — see the csproj's TorchBackend), then CPU.
/// </summary>
public static class DeviceSelector
{
    /// <summary>
    /// c10::DeviceType::XPU. TorchSharp's DeviceType enum has no XPU member, but it passes the
    /// raw value straight through to libtorch.
    /// </summary>
    public const DeviceType Xpu = (DeviceType)12;

    public static Device Best()
    {
        if (cuda_is_available())
        {
            Console.WriteLine("CUDA available — training on GPU.");
            return CUDA;
        }

        if (XpuAvailable())
        {
            OneDnnPrecision.AllowTf32();
            Console.WriteLine("Intel GPU (XPU) available — training on GPU (oneDNN TF32 on, as cuDNN's is by default on CUDA).");
            return new Device(Xpu, 0);
        }

        Console.WriteLine("CUDA not available — training on CPU.");
        return CPU;
    }

    /// <summary>
    /// TorchSharp has no xpu.is_available(); allocating a tensor there fails on a libtorch
    /// built without XPU or a machine without the GPU.
    /// </summary>
    private static bool XpuAvailable()
    {
        try
        {
            using var probe = zeros(1, device: new Device(Xpu, 0));
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }
}
