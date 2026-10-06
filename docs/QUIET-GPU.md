# Quiet-GPU runtime validation

The accepted reconstruction candidate passed three consecutive RTX 2080 Ti checks after the game was exited.

| Display | Combined spatial + temporal p95 range | Proposed gate |
|---|---:|---:|
| 1920x1080 | 0.725-0.840 ms | <=3 ms |
| 2560x1440 | 1.078-1.236 ms | <=4 ms |

Both gates pass for this offline repeated-input microbenchmark. Debug-layer stress parity and full-resolution final-frame parity also pass. The experimental registry now accepts cycle-000003-candidate-03; existing games are not modified.

Each run uses 100 warmups and 1,000 measured repeats per reset and steady-history frame, with steady-history statistics reported separately. Preflight snapshots show about 9 GiB free and 4-6% GPU utilization. See results/quiet-gpu-validation.json and the runtime-check directories for source measurements.

This excludes input/output transfers, CPU fence waits, compilation and engine rendering. The next gate is an integrated D3D12 scene with correct camera/jitter/exposure inputs and whole-frame performance measurement under shared GPU load. Quiet-desktop timing does not replace that gate.
