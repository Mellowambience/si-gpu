# SI-GPU 0.3.0: GPU temporal reconstruction

Implemented and measured on October 5, 2026, using the user's RTX 2080 Ti, driver 617.14, Windows 11 Pro build 26300, and an isolated Python 3.11 environment. The previous 0.2.0 release remains available separately.

## Implemented behavior

The D3D12 sequence runner now combines the existing spatial shader with a new temporal compute pass. Color history stays in two alternating GPU buffers. Low-resolution depth/motion/reactive metadata is also retained in alternating buffers, so the next frame can reproject and compare prior depth without a CPU history transfer.

The temporal pass matches the accepted CPU reference: nearest current depth/motion, current-to-previous motion in render pixels, nearest history-depth lookup, relative-depth rejection, bilinear RGB history, current 3x3 neighborhood clipping, bilinear reactive-mask enlargement and a bounded history blend. First frames and camera-cut flags bypass history. Extreme finite motion is bounded before multiplying by scale.

Each native sequence has fixed dimensions. The bridge rejects an in-sequence resize and instructs the caller to recreate the resources. Independent 1080p and 1440p sequences exercise both sizes. There is no interactive monitor-switch implementation yet.

This is an **offline sequence laboratory**, with input upload and a CPU fence wait per frame. Optional readback captures all frames for parity or only the final frame for profiling. It is not yet an engine adapter, swapchain renderer, neural model, jitter/exposure solution, HDR path or arbitrary-camera upscaler.

## Correctness and synchronization

- 27 unit tests pass, including binary-protocol field preservation, malformed headers, invalid reactive/depth/reset fields and explicit resize rejection.
- The full 15-sequence suite compares 216 output frames against the CPU reference. Both straightforward and tiled shaders pass.
- Three additional eight-frame cases cover history weights 0 and 1 and an exact depth threshold; all 24 compared frames pass.
- The two monitor-size profiles compare final output pixels against the CPU reference for each shader variant.
- D3D12 debug checks were enabled for the small sequence suites and edge cases; the runner fails on debug errors/corruption. No such errors were detected. Performance profiles run without the debug layer.
- Full-suite maximum RGB absolute difference was approximately 6.26e-7, below the 5e-6 tolerance.
- The spatial runner still passes its eight smoke cases after moving shared native helpers into a header.

The suite covers clean/noisy translation, moving foreground silhouettes, fractional motion, depth mismatches, reactive masks, camera-cut resets and extreme motion. Odd output dimensions at 2x/3x/4x exercise partial workgroups. These are procedural, fixed-camera scenes, so parity establishes correct implementation of the existing algorithm, not gameplay-quality superiority.

## Tiled optimization

The straightforward shader independently reads current neighborhood pixels for every output pixel. The accepted tiled shader cooperatively loads an 8x8 output region plus its one-pixel halo into shared memory (100 samples per 64-thread workgroup), then performs neighborhood bounds locally. All threads reach the synchronization barrier, including threads outside a partial output group; bounds checks occur after the barrier.

The simple shader is preserved as gpu/temporal-reference.hlsl. Numerical parity gates were rerun after the optimization. Observed timings improved in this run, but changing desktop interference prevents a controlled universal speedup claim.

## Measured GPU timings

Each monitor profile has two input frames. Frame 0 bypasses history; frame 1 uses it. Each fixed current/prior pair receives 100 warmup and 1,000 timed repeats; history advances once per input frame, not once per repeat. The following numbers are from **frame 1 only**, so a fast reset pass cannot dilute steady-history results.

| Output | Shader | Combined p50 | Combined p95 | Temporal-only p50 |
|---|---|---:|---:|---:|
| 1920x1080 | Straightforward | 2.416 ms | 48.636 ms | 2.236 ms |
| 1920x1080 | Tiled | 0.721 ms | 31.031 ms | 0.250 ms |
| 2560x1440 | Straightforward | 5.651 ms | 48.901 ms | 3.425 ms |
| 2560x1440 | Tiled | 1.429 ms | 49.045 ms | 0.784 ms |

These are sequential, occupied-desktop microbenchmarks. Combined medians need not equal the sum of separate-pass medians. Raw ordered samples, reset medians and shader hashes are retained. Tiled profiles also record NVIDIA snapshots before and after execution. High reported GPU activity and occupied memory support concern about contention, but no scheduling trace has yet established the cause of the long tails.

**The <=3 ms 1080p and <=4 ms 1440p p95 gates remain unpassed.** The full future engine budget must additionally include preparation, required copies, synchronization and measured rendering interaction; this laboratory does not establish frame-rate gain or input latency.

Timing windows cover spatial dispatch plus its resource transition, then temporal dispatch, output UAV synchronization and restoration of the spatial buffer for the next repeat. Input upload, final output readback, CPU waits, shader compilation and engine rendering are excluded. Offline sequence wall time is recorded separately and is not GPU pass time.

At 1440p the declared default-heap buffer payload is approximately 210.94 MiB; at 1080p it is 118.65 MiB. These counts include three low-resolution buffers and three output-resolution color buffers plus weights. They exclude allocation rounding, upload/readback memory, query heaps, driver overhead and future neural activations. They are not measured peak VRAM.

## Reproducibility

Build gpu/build.ps1, install the package into Python 3.11+, then run the temporal suite from the source folder. The Python wheel contains the laboratory code; native executables and shaders remain in the source distribution. Use --project when running elsewhere. --debug requires Windows Graphics Tools to be installed; this machine supported it without any installation or settings change.

Primary outputs:

- results/gpu-temporal-tiled: optimized shader full suite and side-by-side truth/CPU/GPU playback.
- results/gpu-temporal-edges: history-weight and depth-threshold edge cases.
- results/gpu-temporal-profile-tiled: current monitor-size profiles and runtime provenance.
- results/gpu-temporal-profile-reference: earlier straightforward-shader comparison.
- results/gpu-temporal: initial straightforward-shader parity suite.

Older spatial and CPU motion measurements are retained as historical baselines. They do not describe this new combined pass.

## Next milestone

1. Capture GPU queue/scheduling traces and replay with a controlled desktop workload. Do not infer the cause of every long interval from utilization alone.
2. Add a controlled D3D12 engine/sample adapter that consumes buffers without per-frame CPU history readback or forced queue waits. Measure whole-frame effects as well as reconstruction passes.
3. Extend the capture contract to camera matrices, jitter, exposure and color-space conventions. Implement moving-camera depth reprojection before validating arbitrary camera motion.
4. Add render/display resize lifecycle tests, HUD composition and representative particles/thin geometry.
5. Only then evaluate neural reconstruction and adaptive control against the accepted baseline.

API references informing resource synchronization and debug validation: [Microsoft UAV barriers](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_resource_uav_barrier), [Microsoft D3D12 debug info queue](https://learn.microsoft.com/en-us/windows/win32/api/d3d12sdklayers/nn-d3d12sdklayers-id3d12infoqueue). Implementation and measurements are original to this project.
