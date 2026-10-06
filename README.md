# SI-GPU 0.4.0 - Reconstruction Learning Laboratory

SI means **superintelligence**, the long-term project vision. This release implements specialized graphics reconstruction; it does not implement general superintelligence or quantum computing.

Original D3D12/HLSL shaders now perform spatial reconstruction and GPU temporal fusion on the RTX 2080 Ti. Primary target: **1280x720 to 2560x1440**. Secondary target: **960x540 to 1920x1080**. Neural models, engine integration, adaptive control and camera/jitter/exposure handling remain future work.

## Verified on this machine

- 38 tests pass on Python 3.11, including learning gates, model integrity and rollback.
- Straightforward and tiled temporal shaders each pass a 216-frame CPU/GPU parity suite with D3D12 debug checks enabled. An additional 24 edge-case frames pass.
- Full suite maximum RGB absolute difference is about 6.26e-7, within the 5e-6 tolerance.
- Both monitor-size profiles match the CPU reference; the spatial runner still passes its eight smoke cases.
- Three quiet-GPU checks passed: combined steady-history p95 1.08–1.24 ms at 1440p and 0.73–0.84 ms at 1080p. Earlier occupied-desktop runs had p95 tails of about 49 ms and 31 ms; those historical measurements remain recorded.

These are offline microbenchmarks, not gameplay FPS or input latency. The runner uploads input and waits on the GPU per frame; engine integration must remove forced hot-path waits. GPU history itself stays on the device. All inputs are unjittered display RGB with comparable linear depth and fixed sequence dimensions.

Read [the temporal decision report](docs/GPU-TEMPORAL.md). Open [truth/CPU/GPU temporal playback](results/gpu-temporal-tiled/index.html). Earlier [spatial results](docs/GPU-BASELINE.md) and [CPU motion comparisons](results/motion/index.html) are retained as historical baselines.

## Automated learning

The new bounded learning loop trains versioned linear reconstruction candidates, retains replay scenes, checks static and temporal regressions, and optionally accepts passing offline candidates. The delivered lab completed three cycles and accepted a candidate with +0.065 dB audit gain over its initial learned model. Aggressive candidates with about +0.35 dB still-image gains were rejected for temporal regressions. A separate installed-package run verified automatic offline promotion (+0.063 dB audit gain).

Open [the learning dashboard](results/learning/index.html) and read [the learning report](docs/LEARNING-LAB.md). The accepted runtime candidate now passes three quiet-GPU validation runs: combined p95 0.73-0.84 ms at 1080p and 1.08-1.24 ms at 1440p. Earlier GPU deferrals are retained as history. No persistent service is running. Training uses original synthetic pairs and the existing linear filter; neural and engine-capture learning remain future work.

```powershell
./.venv/Scripts/python.exe -m siware.learning --lab results/new-learning init
./.venv/Scripts/python.exe -m siware.learning --lab results/new-learning run --cycles 3 --max-seconds 300 --auto-promote-lab --work work/new-learning
./.venv/Scripts/python.exe -m siware.learning --lab results/new-learning status
```

Use a new lab path for init; the bundled results/learning already contains a working lab. Offline promotion and experimental GPU runtime promotion are separate. Read the report before runtime promotion; no game integration is provided.

## Build and run on Windows

Use Python 3.11+, Visual Studio C++ Build Tools and a Windows SDK. No CUDA Toolkit is required. Optional --debug checks need Windows Graphics Tools; it was already available on this PC.

```powershell
py -3.11 -m venv .venv
./.venv/Scripts/python.exe -m pip install .
./gpu/build.ps1
./.venv/Scripts/python.exe -m unittest discover -s tests -v
./.venv/Scripts/python.exe -m siware.gpu_temporal --debug --output results/gpu-temporal --work work/gpu-temporal
./.venv/Scripts/python.exe -m siware.gpu_temporal --profile-only --output results/gpu-temporal-profile --work work/gpu-temporal-profile --repeats 1000 --warmups 100
```

If py -3.11 cannot discover your interpreter, invoke Python 3.11 by its full path to create the environment. Run in the extracted source folder or pass --project explicitly. The wheel contains laboratory Python code; native executables and shaders remain in this source distribution. Native runners were built for Windows x64 from the bundled source.

Each profile has a reset frame and a steady-history frame, with 100 warmup and 1,000 measured repeats per frame. Current and prior inputs are fixed within a profile repeat; history advances once per input frame. Spatial, temporal, combined and reset/steady samples are reported separately. Timing excludes upload, readback, compilation, CPU fence waits and engine rendering.

The versioned binary protocol validates float32 color, depth, motion and reactive fields and boolean resets. Dimensions remain fixed per sequence; resource recreation is required on resize. Reference and tiled shaders are both retained.

## Next milestone

Capture controlled GPU queue/scheduling traces, integrate the passes into a D3D12 scene without forced per-frame CPU history transfers/waits, then validate camera matrices, jitter, exposure, color space, resize and HUD composition. Neural reconstruction and adaptive control follow those gates.

Historical SI-WARE roadmap/calendar files are planning artifacts, not current automation or performance evidence. No events, reminders, drivers or OS settings were changed. The older release archives remain available separately.

