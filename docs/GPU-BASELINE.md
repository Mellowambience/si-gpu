# SI-GPU 0.2.0: GPU baseline decision report

Built and measured on October 5, 2026, on the user's RTX 2080 Ti (11 GiB), NVIDIA driver 617.14, Windows 11 Pro build 26300, i9-9900K and 64 GiB RAM. Original SI-WARE v0.1.1 is preserved separately.

## What is implemented

Original D3D12/HLSL compute code performs half-pixel bilinear reconstruction and the learned polyphase linear residual filter. It uses GPU-resident input/weights and an output UAV. Source data is uploaded once before the timed batch and output is read back once after it. No neural runtime, CUDA Toolkit, engine hook, CPU emulation or quantum component is needed for this pass. A hardware D3D12 adapter is required; WARP/CPU fallback is explicitly disabled.

The native runner selects the compatible nonsoftware adapter with the most dedicated video memory, compiles the shader, and records per-dispatch timestamps, a UAV barrier and 100 warmup dispatches before 1,000 measured large-resolution samples. Resource transitions, uploads, readback, model fitting and shader compilation are outside the per-dispatch timing interval. Timestamp intervals include GPU scheduling effects/interference and are not a measurement of an isolated kernel's intrinsic cost.

The Python harness trains a 2x reference model, generates deterministic float32 inputs, executes the native runner, reads the output and compares all RGB pixels against Pillow/NumPy. It validates both bilinear and learned modes on random and constant tiny inputs, irregular dimensions and scales 2/3/4, then 960x540 to 1920x1080 and 1280x720 to 2560x1440. It rejects max absolute error above 3e-6. All twelve cases pass in both recorded runtimes.

This is a spatial reconstruction microbenchmark. Temporal reconstruction remains CPU reference code. There is no swapchain, interactive engine scene, neural model, adaptive controller or measured FPS/input-latency benefit in this release.

## Measured results and failed speed gate

| Output / learned linear mode | Initial run p50 | Initial run p95 | Python 3.11 replay p50 | Python 3.11 replay p95 |
|---|---:|---:|---:|---:|
| 1920x1080 | 0.161 ms | 20.070 ms | 0.160 ms | 16.495 ms |
| 2560x1440 | 0.288 ms | 5.378 ms | 0.297 ms | 22.981 ms |

Each large case has 100 warmups and 1,000 measurements. The initial run used Python 3.10/NumPy 2.2.6; the replay used Python 3.11/NumPy 2.4.6. Those two runs do not isolate Python-version effects: the timed shader is native, and desktop GPU conditions changed. Recorded NVIDIA snapshots show substantial other activity/occupied memory. The long tail may include interference, preemption and scheduling; its precise cause is not yet isolated. No application was stopped or system setting changed.

The proposed <=4 ms 1440p and <=3 ms 1080p p95 targets are **not passed under the recorded conditions**. Even a passed isolated spatial target would not pass the eventual complete reconstruction budget, which must include temporal fusion, preparation, copies and synchronization in an engine.

At 1440p, input + output + weights contain about 70.31 MiB of GPU-buffer payload. This excludes driver allocation rounding, GPU runtime memory, upload/readback heaps, query memory and any future temporal/neural buffers. It is not peak VRAM. Raw timing samples and pixel parity results are in results/gpu and results/gpu-py311. CPU reference timing is a single run and must not be used as a matched speedup benchmark.

## Motion lab findings

The old wrapping translation test allowed image-edge reconstruction error to propagate into its small crop. The replacement generates truth by moving a window across a larger canvas without wrap, and separates a history-safe interior from the boundary region. This is an evaluation correction and diagnosis, not a claim that every temporal artifact is fixed.

Three unseen seeds, translation and a foreground/background silhouette, clean and noisy input, and sixteen frames each give 192 evaluated frames. Full-resolution depths and motion provide visibility/disocclusion masks. Comparison GIFs show truth, spatial and temporal output side by side. Interior metrics, boundary metrics and newly revealed-surface errors are all retained.

Clean integer translation in the history-safe interior is stable (below 1e-12 aligned-error-change MSE in the regression test). Noisy cases show lower temporal error fluctuation. Boundary errors remain nonzero. In the seed-200 silhouette case, newly revealed-background MSE is the same for spatial and temporal output, consistent with history being rejected rather than adding recovered detail.

These are original synthetic, fixed-camera, unjittered display-RGB cases. They do not validate moving-camera depth transforms, HDR, exposure, particles, HUD or gameplay. The diagnostic is custom and does not replace visual review or perceptual evaluation.

## Validation and next decision

21 CPU tests pass on Python 3.10 and 3.11. The native program compiles with MSVC 2022 x64 against the installed Windows SDK. Hardware parity passes all twelve GPU cases at both monitor targets. An isolated Python 3.11 environment installs the package. Runtime dependencies and source hashes are retained with the release.

Next: collect controlled GPU scheduling traces and a repeatable quiet-desktop replay, then add GPU temporal fusion with CPU parity on the motion suite. Follow with camera/jitter/exposure captures and one engine scene. Keep neural training and learned control behind those data and integration gates.

Implementation references: [Microsoft D3D12 timestamp queries](https://learn.microsoft.com/en-us/windows/win32/direct3d12/timing), [Microsoft root signatures](https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signatures-overview). Source and design decisions are original; these references document the graphics API behavior.
