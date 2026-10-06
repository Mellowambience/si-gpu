# SI-WARE roadmap — October 6 to November 30, 2026

## Outcome and constraints

Ship an **experimental** reconstruction lab, then a one-scene GPU integration if the quality and speed gates pass. Work budget: roughly 10 hours/week, 80 hours total. This is a planning assumption, not a delivery guarantee. Your previously reported RTX 2080 Ti / i9-9900K / 64 GB Windows PC is the first validation target; it has not been inspected or benchmarked in this build.

Vendor-neutral means an architecture with multiple backend options and published minimum requirements. It does not mean every computer can run every quality preset. Quality, latency, memory and supported-device evidence decide the eventual claims. Frame generation stays outside the first release.

| Dates | Milestone | Build | Acceptance gate |
|---|---|---|---|
| Oct 6–12 | M1: Reference lab | Run current CPU prototype, review data contract, record local environment | Tests pass locally; demo reproducible; baseline/model measurements retained. Initial code and host verification completed Oct 5. |
| Oct 13–19 | M2: Evaluation data | Capture or procedurally render paired native/low-res sequences; motion/depth/jitter metadata | At least 3 independent scenes and 300 paired frames; whole scenes separated for evaluation; rights recorded; fixed degradation protocol. |
| Oct 20–26 | M3: Neural candidate | Small residual CNN with subpixel output; train on owned/permitted data | Checkpoint and reproducible evaluation; aim for >=0.5 dB held-out gain over bilinear without obvious ringing. Benchmark bicubic/Lanczos too; reject regressions rather than claim superiority. |
| Oct 27–Nov 2 | M4: Temporal fidelity | Add jitter/exposure handling, camera-cut metadata, reprojection depth handling | Translating-camera, thin-geometry, disocclusion, particles and HUD sequences pass visual review; no major trails; report temporal error against ground truth and single-frame baseline. |
| Nov 3–9 | M5: GPU path | Choose supported inference backend; port reconstruction; profile on 2080 Ti | Publish p50/p95 GPU pass time, peak VRAM and environment at 720p->1080p first; aspirational <=4 ms p95. Profile 720p->4K separately; do not extrapolate. |
| Nov 10–16 | M6: One-engine scene | Select engine with buffer access; integrate deterministic sample scene | Repeatable captures with upscaling on/off; correct motion, depth, jitter and resize reset; HUD composited after reconstruction. No arbitrary-game injection. |
| Nov 17–23 | M7: Portability | Test a second actual GPU/backend where available; fallback behavior | Publish tested-device matrix with versions; graceful unsupported-device behavior. If hardware is unavailable, mark support unverified and postpone claims. |
| Nov 24–30 | M8: Experimental release | Package source, model card, captured demo, measurements, integration instructions | Clean-machine install on at least the first target; artifact review completed; disclose failed gates and restrictions. Release offline lab if GPU/engine gates remain blocked. |

## Weekly rhythm

Proposed sessions: Tuesday 2h implementation, Thursday 2h implementation, Saturday 3h experiments, Sunday 2h evaluation/documentation, Monday 1h review. Work-session times in the included calendar are editable proposals: evenings Tue/Thu/Mon, Saturday afternoon, Sunday late afternoon, all America/New_York. No calendar events were inserted automatically.

ChatGPT milestone check-ins: Monday evenings around 7 pm, Oct 12 through Nov 30 (8 occurrences). These reminders do not run GPU jobs, operate your PC, or imply unsupervised development.

## Dependency gates and scope control

M2 precedes neural claims; M3 and M4 precede meaningful GPU comparisons; M5 precedes integration promises; M6 precedes gameplay demonstrations; M7 precedes multiple-device compatibility claims. If a gate fails, use the next week's first build block to diagnose it and move dependent dates. Keep the measured working baseline.

At 80 hours, the realistic deliverable is a research prototype or a narrow experimental integration. A broadly supported production upscaler requires additional engineering and device testing beyond this plan.

## First three actions

1. Extract the ZIP and run tests/demo on your Windows PC.
2. Save its environment, measurements and one gameplay crop comparison. Inspect text, foliage, thin lines and particles rather than only a still image.
3. Choose the controlled scene capture source and establish paired high/low-resolution data with metadata. A screen recording alone lacks engine depth and motion vectors.

## Work completed in this build

- Original offline CPU reconstruction code, model training and inference, CLI commands, HTML demo.
- Depth-rejected motion reprojection, reactive masks, neighborhood clipping, resets.
- Ten tests passing in the execution environment.
- Synthetic held-out mean +0.398 dB gain versus bilinear; neural/GPU/gameplay evidence still pending.
- Importable calendar and eight scheduled milestone reminders.
