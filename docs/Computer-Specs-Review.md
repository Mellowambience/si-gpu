# Computer review for SI-GPU

Read-only inspection on October 5, 2026. No settings, drivers or installed applications were changed.

| Component | Observed specification |
|---|---|
| CPU | Intel Core i9-9900K, 8 cores / 16 threads, reported base clock 3.60 GHz |
| GPU | NVIDIA GeForce RTX 2080 Ti, 11,264 MiB (11 GiB) VRAM, CUDA compute capability 7.5 |
| NVIDIA driver | 617.14 |
| RAM | Four 16 GiB TEAMGROUP-UD4-3200 modules; 64 GiB installed; configured speed reported as 2400 |
| Motherboard | MSI MPG Z390 GAMING EDGE AC (MS-7B17) |
| OS | Windows 11 Pro, 64-bit, build 26300 |
| Main monitor | ASUS VG32AQL1A, preferred/native mode reported as 2560x1440 |
| Secondary monitor | MSI Optix AG32C, preferred/native mode reported as 1920x1080 |
| NVMe SSD | WDS100T3XHC-00SJG0, approximately 1 TB |
| Internal HDD | ST4000DM004-2CV104, approximately 4 TB |
| USB storage | Seagate Portable, approximately 4 TB |

Volume free space: C approximately 117.4 GiB, D approximately 3223.5 GiB, E approximately 1282.4 GiB. Storage health reports Healthy; this was not a full disk diagnostic. Monitor preferred modes were verified; the video-controller query reports one 1920x1080 mode at 164 Hz and cannot establish both monitors' active resolutions/refresh rates.

## Development environment

Python launcher discovers Python 3.13, Python 3.10 and an uv-managed Python 3.11.15. The `python` command resolves to Python 3.10, while `py` defaults to 3.13. Use an explicit interpreter and isolated environment for the project rather than relying on those different defaults.

Git and CMake are present. Visual Studio C++ tooling is detected in 2022 Build Tools, 2019 Build Tools and 2019 Community installations. Windows SDK include directories exist for versions 10.0.18362.0, 10.0.19041.0 and 10.0.26100.0. This is inventory evidence, not a successful compiler/shader build.

No `nvcc` was found on PATH and no CUDA Toolkit was found in its standard installation folder. A custom installation remains possible. NVIDIA-SMI's CUDA 13.4 display is the driver's supported CUDA level, not proof of an installed development toolkit. [NVIDIA explanation](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html)

## Assessment

This is a reasonable starting machine for a compact reconstruction model and the proposed 1440p GPU demonstration. That is an engineering assessment, not proof that the proposed timing gates will pass. The 11 GiB VRAM capacity makes model size, training crop size and concurrent application use important.

At inspection, NVIDIA-SMI reported approximately 9.6 GiB occupied and 1.2 GiB free, with 12–14% GPU utilization and 48–49 C temperature. This is a transient snapshot; it does not indicate sustained rendering load or identify which application owns the memory. Windows WDDM process accounting did not supply reliable per-process memory totals. Start benchmarking from a controlled workload and recheck free VRAM before training.

RAM part numbers suggest a 3200-class kit, but the actual reported configuration is 2400. Investigate the BIOS memory profile and module specification separately if tuning is desired; no BIOS changes were made or recommended as necessary for this prototype.

Use the NVMe drive for active code, training caches and compact captures; keep larger archives on the other drives. With roughly 117 GiB free on C, measure capture sizes before collecting large uncompressed sequences.

The first technical step is a D3D12 build and GPU timestamp smoke test, followed by a compact-model inference compatibility test. Do not assume Windows ML's downloadable NVIDIA provider supports this card: Microsoft's current catalog specifies RTX 30-series or later for that provider. [Microsoft provider requirements](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)

No hardware purchase is needed to begin the planned lab and baseline development. Revisit upgrades only after measurements identify a constraint.
