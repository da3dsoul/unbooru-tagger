# Engineering and hardware findings

These are measured findings about running TorchSharp training on an NVIDIA RTX A4000 (16 GB) and an Intel Arc
Pro B70 (32 GB). The how-to versions live in [docs/TRAINING.md](../docs/TRAINING.md).

## Moving training to the Intel Arc Pro B70 (2026-10-24)

### Setting it up

- **Stock libtorch works.** Stock TorchSharp 0.107.0 runs on the B70 against the libtorch in PyTorch's
  `torch==2.10.0+xpu` wheel. 2.10.0 is the libtorch TorchSharp 0.107 is built against, and the C++ ABI matches.
- **The device enum.** TorchSharp's `DeviceType` enum has no XPU, so the device is `(DeviceType)12`.
  `DeviceSelector` probes CUDA, then XPU, then the CPU.
- **TF32.** Only the legacy `at::Context::setAllowTF32OneDNN(true)` reaches the XPU convolutions, called through
  libtorch's exported symbol (`OneDnnPrecision`). The newer `setFloat32Precision` reads back as set but has no
  effect. TF32 took the tower from 203 to 138 ms/step at batch 10.

### Speed against the A4000

| Measurement | B70 | A4000 |
|---|---|---|
| Run 2's configuration, full training step (TF32) | 158 ms | 278 ms |
| Same, without TF32 | 219 ms | |
| Run 3 (fused-3) | 172 ms | |

That makes the B70 about 1.8x the A4000, or about 1.6x with fused-3.

### Memory ceiling

- fused-2 fits about batch 32 (batch 30 = 27.9 GiB).
- fused-3 holds 17.9 GiB at batch 10.
- Bigger batches don't raise images per second; the card is compute-bound at batch 10.

## Failure modes on the B70, and their fixes

| Symptom | Cause | Fix |
|---|---|---|
| Over-allocation silently spills VRAM into host RAM and thrashes over PCIe. | Implicit USM migration in the Intel compute runtime. | `NEOReadDebugKeys=1 EnableSharedSystemUsmSupport=0 EnableImplicitMigrationOnFaultableHardware=0`. Over-allocation then fails with `UR_RESULT_ERROR_OUT_OF_RESOURCES`. Taken from the maintainer's earlier Arc LLM experiments, set there after two host-wide OOM kills. |
| After Ctrl+C mid-step, the next trainer hung in `Checkpoint.Load` → `Tensor.copy_` at 100% CPU, once for 7 hours. The kernel log showed an xe "Engine memory CAT error, class=bcs". | Copies were still in flight when the process died. | Commit `829885f`: the first Ctrl+C finishes the step, saves and exits; a second one quits at once. |
| The copy engine (bcs) faulted mid-run: the same xe CAT error at nearly the same GPU address on 10-24, 10-27 (x4), 10-31 (x3) and 11-01 (x2), and the trainer hung in `backward()`. | The copy engine itself. It happened with or without pinned staging buffers. The same signature is reported on other B70s (intel/torch-xpu-ops#5867, vllm#61300). Compute runtime 26.05.37020 here. | `OverrideCopyOffloadMode=0` (a NEO key; needs `NEOReadDebugKeys=1`) moves every copy to the compute engine: bcs cycles 0, same speed (204 ms/step). `UR_L0_USE_COPY_ENGINE=0` and `ForceCopyOperationOffloadForComputeCmdList=0` don't stop bcs use. Checkpoints are also written to a staging directory and moved in once complete (`StagedSave`). |
| Unexpected multi-second steps; costs superlinear in a varying dimension. | **oneDNN shape churn.** Every new input shape compiles new kernels (about 2.5 s for the run-4 tower). The primitive cache holds about 7 tower shapes. | Keep every shape that reaches a convolution constant: pad, don't trim. Glimpse batches padded to a fixed size: 426 → 237 ms/step. |
| VRAM ratchets up (about 2 GB a minute, or 12 → 24 GB in 10 minutes), then xe "VM worker error: -12" and `UR_RESULT_ERROR_OUT_OF_RESOURCES`. | Tensors whose sizes change nearly every step make the caching allocator reserve new blocks. The suspected ops are `masked_select` and `index_put_` in the locality table update. | Pad rows to fixed widths: locality rows to 32 (still died after 47 minutes); run 4's localization pair rows to multiples of 64 (2026-11-03). Build labels with fixed-shape `where`/`gather`. |
| Any upload from ordinary host memory blocks until the whole device queue drains (147 ms behind 146 ms of queued work); so does any read-back. | Pageable memory on this backend. | Upload from pinned memory with `non_blocking` (`DeviceUpload`), keep the pinned source until the device has finished, and read losses back only every N steps. |
| The trainer slowed to about 3 s/step for about 4 hours, with the B70 about 7% busy. | Host RAM pressure: swap full. `/tmp` is a tmpfs, so scratch builds there count against RAM. | A graceful stop and restart restored 223 ms/step. Keep host RAM free. |
| A CUDA evaluation launched from the XPU trainer threw "Cannot have both XPU and cuda". Snapshots 130K-180K failed silently. | It inherited `LD_LIBRARY_PATH` pointing at the XPU libtorch. | `unset LD_LIBRARY_PATH` in every CUDA launcher started from XPU scripts. Check snapshot directories for `evaluation.failed`. |
| A new trainer started seconds after another exits fails with `UR_RESULT_ERROR_OUT_OF_RESOURCES`. | Not established. | Wait before relaunching. |

## Speed work

| Change | Effect | Where measured |
|---|---|---|
| Sparse localization loss (about 100 pairs per image instead of 25K) | 1,413 → 468 ms/step | old tower |
| `BatchPrefetcher` with offset-sorted parallel reads | 48 → 450-590 img/s; step waits about 1 ms on data | RAID5 |
| `ForwardPooled`: pool the full map without building it (exact for 2x nearest upsampling) | about 1.5-2% | B70, run 4 |
| Merging mosaic canvases into the batch's forward | **slower** (241 vs 230 ms/step) | B70; not used |
| Channels-last via permute strides (10-28) | no faster (203.9 vs 203.6); the "different outputs" finding was later traced to the unseeded XPU RNG | B70 |
| `--bf16`: bfloat16 compute from fp32 weights cast per call (TorchSharp has no autocast) | tower forward + backward 204 → 164 ms; run 4 step 223 → 199 ms; 6,000-step A/B kept mAP (0.576 vs 0.575) and gold pointing (32.3% vs 32.6%) | B70 |
| ConvNeXt's 1x1 convolutions as one matmul (`MixedPrecision.Rows`); oneDNN's 1x1 weight gradient was about 7x the forward | tower forward + backward 137 → 127 ms | B70 |
| Channels-last tower with the group norm as a layer norm on the NHWC view (same values and gradients, tested) | run 5 228 → 205 ms/step | B70 |
| Run 5's speed work applied to run 4 | 221 → 142 ms/step, same recipe; a 12K arm reached the same mAP | B70 |

**The step profile** (2026-10-28, 200 synchronized steps): backward 63%, forward 25%. The tower is about all of
it, and it is memory-bandwidth bound: elementwise ops run at about a fifth of the B70's bandwidth, and the
matmuls use a small fraction of its compute. That is why int8 wouldn't help. The tools are `UNBOORU_PROFILE_STEPS`
and `benchmark-tower`.

## TorchSharp pitfalls

- **No refcount-driven freeing.** A tensor lives until its `DisposeScope` ends. Module forwards need their own
  scope, with `MoveToOuter` on the result; otherwise evaluation memory balloons (13.2 GB → 4.9 GB after the fix).
- **GroupNorm affine parameters were left uninitialized.** That is NaN on the CPU and zeros on CUDA. They are
  now initialized to 1/0; run 2 started before this fix.
- **No `Module.to(device)` in this version.** Modules are built on their device.
- **`HashCode.Combine` is randomized per process.** Use `DeterministicSeed`.
- **`manual_seed` doesn't seed the XPU generator.** Create random weights or inputs on the CPU and move them.

## Splitting work across two GPUs

- **Offloaded evaluation.** Training runs on the B70. Epoch and snapshot evaluations run as separate processes on
  the A4000 (`--evaluation-command`, flock on a lock file). A CUDA and an XPU libtorch can't share one process.
  Scores on the two GPUs agree to about 3 decimals (tf32 on both).
- **Host memory.** An A4000 evaluation process holds about 13.5 GB RSS in host memory (run 4's tower, 20K held-out
  images with score dumps, measured 2026-11-03) to about 22 GB (run 5 glimpse checkpoints, pooling notes,
  2026-11-02).
- **GPU memory.** An A/B training arm on the A4000 reserved 14.5 GB. On the B70, run 4's trainer ratcheted from 11.9
  to 23.7 GB in its first 10 minutes on build 37 (`review/vram-run4.log`) and died 40 minutes in (xe "VM worker
  error: -12", `UR_RESULT_ERROR_OUT_OF_RESOURCES`, 2026-11-03 12:20). The localization pair rows were as wide as
  the batch's widest image, so their size changed nearly every step (the widest seen: 320), and each new size made
  the caching allocator reserve blocks it never returned. Rows padded to multiples of 64 (weight-0 padding, the
  loss unchanged; build 38) hold a flat 25.3 GB (`review/vram-run4-b38.log`), at about +3% step time. Run 5's
  locality term had died the same way on 2026-11-01. Padding didn't bound the peak: a pointing A/B arm with
  `--localization-weight 0.6` on build 38 reached 29.4 GB (the widest row 788), since some images carry hundreds of
  certain negatives (a character with many context targets, an excludes rule over a whole family). Build 39 caps an
  image's labelled pairs at 192 and a mosaic tile's pairs from other tiles at 128. It keeps every positive and samples
  the negatives, scaling their weights by all ÷ kept so the expected loss is unchanged. Run 4 runs it from 2026-11-03
  14:34 at 144 ms/step.
