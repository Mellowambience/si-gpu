# SI-GPU 0.4.0: automated learning laboratory

Implemented and verified October 5, 2026. This release prioritizes the requested learning loop; the previously planned engine/camera integration remains pending. Older release archives are preserved.

## Working loop

The new command-line module generates original paired procedural examples, retains original training scenes plus a bounded replay of recent examples, fits versioned reconstruction candidates, evaluates them, nominates a winner and optionally promotes it for offline laboratory use. State persists between bounded runs.

It currently trains the existing **linear residual filter**, not a neural network or an appearance generator. Training is CPU-only. It does not capture games or screenshots, use cloud compute, start a daemon, create scheduled jobs or change a game's rendering settings.

There are two accepted-model pointers:

- accepted_lab: an offline reconstruction model that passed the declared laboratory gates.
- accepted_runtime: a separately gated experimental GPU model. It was initially null under high GPU load; the quiet-GPU update below records its passing promotion. It does not represent a production game integration.

Each model is immutable and recorded with a SHA-256 checksum. Promotion swaps a registry pointer rather than overwriting model weights. Rollback restores the prior pointer after verifying the old model. A single-writer lock prevents overlapping mutations, and registry/report writes replace completed temporary files. Failed cycles cannot supply passing nominations. Changed gate configurations invalidate previous nominations.

## Data and gates

Training starts with seeds 0-11. Each cycle adds eight new scenes from a separate seed range above 10000. The original twelve anchors remain; up to 32 recent scene IDs are replayed. Cached pairs are original synthetic geometry generated locally, with provenance and dataset hashes in experiment manifests.

Validation scenes 100-111 select candidates; regression scenes 200-211 check older-scene quality; moving checks use seeds 300-301. Candidate ridge values are explicit configuration. The latest default explores eight values, including conservative alternatives to the sharpest filters.

The best candidate among passing development checks is frozen before a release-audit batch is evaluated. The final code reserves a fresh contiguous audit block per cycle below the training seed range, to reduce repeated test-set tuning. The archived first demonstration used the original predeclared audit seeds 5000-5007; future cycles use their derived untouched blocks. Exhausted audit pools require a new documented split. Different seeds from one generator are still a narrow synthetic holdout, not independent gameplay domains.

Lab gates are:

- Mean validation PSNR gain at least 0.01 dB against the accepted model; no static scene regression greater than 0.05 dB.
- No material temporal regression: at most 2% additional reconstruction/disocclusion MSE and 5% additional aligned-error-change diagnostic, with an absolute 1e-10 numerical floor.
- Interleaved 30-sample CPU inference p95 at most 1.5 times the accepted reference on the small test input. This is a coarse cost check, not monitor-resolution GPU evidence.
- Mean release-audit gain at least 0.01 dB and the same scene-regression bound.

GPU evidence is optional and separate. The RTX-targeted preflight defers evaluation if NVIDIA status is unavailable, utilization exceeds 20%, or free VRAM is below 768 MiB. If it proceeds, it requires model-specific CPU/GPU parity and combined steady-history p95 <=3 ms at 1080p and <=4 ms at 1440p. Profile windows exclude offline transfers, compilation and CPU waits. Buffer payload is recorded but is not measured peak VRAM; the eventual engine/memory acceptance gate remains separate.

No rejected candidate becomes accepted automatically. --auto-promote-lab accepts only passing offline candidates between completed cycles. Runtime promotion is an explicit separate command and refuses candidates without passing runtime evidence. The later quiet-GPU update promotes the experimental registry pointer. No game integration was activated.

## Results retained

The primary lab ran three cycles, training 18 candidate models (five, five and eight). The first two cycles kept the original model. Their most aggressive candidates improved validation still-image PSNR by about 0.347 dB but failed temporal checks, with diagnostic fluctuation ratios reaching about 1.64. This is a real example of the guard rejecting an appealing still-image gain.

Cycle 3's milder ridge=0.008 candidate improved validation PSNR by about 0.060 dB and release-audit PSNR by **0.065 dB** over the initial learned baseline. It passed the old-scene, temporal and CPU-cost gates. The lab pointer was promoted, rolled back to baseline and promoted again; model files stayed unchanged. The delivered accepted offline model is cycle-000003-candidate-03.

Runtime checks were deferred under high GPU utilization. The final corrected preflight records deferred_busy_or_low_memory, and accepted_runtime was initially null. Earlier unknown-state records are retained; the initial parser did not handle NVIDIA's unit suffixes correctly. A regression test now verifies that busy snapshots containing '%' and 'MiB' defer before any GPU dispatch.

An independent fresh lab, run through the installed Python package with --auto-promote-lab, completed one cycle and automatically accepted a passing candidate with **0.063 dB** release-audit gain. This demonstrates actual automated promotion, not only a mocked test. That evidence is copied into results/learning-auto-proof.

**38 tests pass**, covering model/protocol behavior, checksums, stale gates, failure rejection, writer locking, runtime refusal, promotion and rollback. Package installation into the isolated Python 3.11 environment succeeds. The learning addition does not change the native shaders or claim a new GPU speed improvement.

## Commands

Run in the source folder after installation. Existing results/learning already contains the demonstrated lab; do not init that same directory again. Use a new lab path for a clean experiment.

```powershell
./.venv/Scripts/python.exe -m siware.learning --lab results/new-learning init
./.venv/Scripts/python.exe -m siware.learning --lab results/new-learning run --cycles 3 --max-seconds 300 --auto-promote-lab --work work/new-learning
./.venv/Scripts/python.exe -m siware.learning --lab results/new-learning status
```

GPU checks, later promotion and rollback:

```powershell
./.venv/Scripts/python.exe -m siware.learning --lab results/learning check-runtime cycle-000003-candidate-03 --project . --work work/runtime-check
./.venv/Scripts/python.exe -m siware.learning --lab results/learning promote cycle-000003-candidate-03 --scope runtime
./.venv/Scripts/python.exe -m siware.learning --lab results/learning rollback --scope lab
```

The runtime promotion command intentionally fails unless its evidence passes. --gpu-gates requests runtime evaluation during a cycle when a candidate qualifies; it does not bypass preflight. Bounded run controls allow 1-10 cycles and 30-1800 seconds overall. Training data has a 512 MiB configured cache budget; that is not a whole-project storage budget. A PAUSE file in the lab directory blocks the next cycle. A forced process termination can leave a writer.lock that needs inspection; it is not automatically stolen.

Open results/learning/index.html for cycle reports and an accepted-model example. Reports retain training/evaluation roles, configuration hashes, per-scene gains, temporal failures, timings and candidate checksums. Changing search settings is legitimate development, but loosening gates after seeing failures does not demonstrate improvement.

## Remaining work

Add an integrated capture source with native references and camera metadata, then compact neural reconstruction training. Build a genuine idle service with gameplay/CPU/GPU/thermal awareness, bounded queues and session-boundary activation only after its evaluator is trusted. This release supplies bounded repeated experiments and optional offline promotion; continuous background scheduling is not enabled.

The small synthetic gains do not establish DLSS equivalence, general superintelligence, or production quality. Continual-learning research motivates replay to preserve earlier behavior, but our replay policy needs broader scene testing. [Experience Replay for Continual Learning](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html)


## Quiet-GPU validation update

After the user exited the game, preflight showed roughly 4-6% GPU utilization and 9 GiB free VRAM. Three consecutive model-specific GPU checks passed debug-layer parity and the two monitor timing gates. Combined learned-spatial/temporal steady-history p95 ranges were **0.725-0.840 ms at 1920x1080** and **1.078-1.236 ms at 2560x1440**.

The candidate cycle-000003-candidate-03 is now the accepted experimental runtime model as well as the accepted offline model. Each monitor check used 100 warmups and 1,000 measured repeats per input frame; the reported steady-history samples exclude reset frames. Raw data and snapshots remain in the cycle's runtime-check directories, with a consolidated results/quiet-gpu-validation.json.

The quieter results are consistent with earlier game/desktop contention affecting timing. They do not identify every prior long interval's cause, and they do not establish timing while sharing the GPU with a real engine. Pass timings exclude uploads/readback, CPU waits, compilation and scene rendering. Full engine integration, neural models and persistent background scheduling remain pending.
