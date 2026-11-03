using TorchSharp;
using static TorchSharp.torch;

namespace UnbooruTagger.Training.Model;

/// <summary>
/// Host-to-device copies that don't hold the CPU. A copy from ordinary (pageable) memory waits for everything already queued
/// on the device: on the B70, uploading a batch behind ~146 ms of queued work held the CPU 147 ms, so every upload in a
/// training step drained the queue and left the device idle while the CPU set up the next work. From pinned memory the copy
/// is queued like any other op (3.7-12.7 ms of CPU for a 7.8-31 MB batch, most of it the copy into pinned memory).
/// <para>
/// The pinned staging copies are kept until the device has certainly finished with them: until <see cref="Release"/>, which
/// the caller calls right after something that waits for the device (training: reading the losses back every 5 steps), or
/// until <see cref="MaxPendingBytes"/> of them pile up, when this waits for the device itself. Freeing them as soon as the
/// copy was queued looked safe in <c>benchmark-tower --uploads</c>, but after ~25 minutes of training (2026-10-31 23:28) the
/// B70's copy engine faulted reading host memory (xe: "Engine memory CAT error, class=bcs"), the engine was reset and the
/// trainer hung in backward().
/// </para>
/// </summary>
public static class DeviceUpload
{
    /// <summary>Pinned staging copies held before a release is forced.</summary>
    public const long MaxPendingBytes = 2L << 30;

    private static readonly List<Tensor> Pending = [];
    private static long _pendingBytes;
    private static readonly object Lock = new();

    /// <summary><paramref name="host"/> on <paramref name="device"/>, copied without waiting for the device's queue.</summary>
    public static Tensor Upload(this Tensor host, Device? device)
    {
        if (device is null || device.type == DeviceType.CPU)
            return device is null ? host : host.to(device);
        if (host.device_type != DeviceType.CPU)
            return host.to(device);
        return UploadPinned(host.pin_memory().DetachFromDisposeScope(), device);
    }

    /// <summary>
    /// A pinned host tensor on <paramref name="device"/>, copied without waiting for the device's queue. Takes ownership of
    /// <paramref name="pinned"/>: it is freed at the next <see cref="Release"/>, once the device has certainly read it.
    /// </summary>
    public static Tensor UploadPinned(Tensor pinned, Device device)
    {
        var uploaded = pinned.to(device, non_blocking: true);
        bool full;
        lock (Lock)
        {
            Pending.Add(pinned);
            _pendingBytes += pinned.NumberOfElements * pinned.ElementSize;
            full = _pendingBytes > MaxPendingBytes;
        }
        if (full)
        {
            // Wait for the device: a read back of a new device tensor is queued behind every copy so far.
            using (var probe = zeros(1, device: device))
            using (var back = probe.cpu())
                _ = back.NumberOfElements;
            Release();
        }
        return uploaded;
    }

    /// <summary>
    /// Frees the pinned staging copies. Only call it right after something that waited for the device to finish everything
    /// queued so far (a read back), or a copy may still be reading one.
    /// </summary>
    public static void Release()
    {
        List<Tensor> done;
        lock (Lock)
        {
            done = [.. Pending];
            Pending.Clear();
            _pendingBytes = 0;
        }
        foreach (var tensor in done)
            tensor.Dispose();
    }
}
