# v0.1.1 refinement and research

## Research basis

AMD describes paired selection of depth and motion from the nearest-camera surface in a 3x3 neighborhood. That supports avoiding independent averaging of metadata across silhouettes. This release uses a simpler nearest-sample policy, not AMD's dilation/reconstruction implementation. Engine-scale reprojection still needs camera transforms, jitter and exposure handling.

Source: [AMD temporal super-resolution reference](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-temporal/).

Pillow documents bilinear, bicubic and Lanczos resampling. The expanded comparison applies those filters to floating RGB channels, clips ringing excursions to [0,1], and compares the same targets and output sizes.

Source: [Pillow Image reference](https://pillow.readthedocs.io/en/stable/reference/Image.html).

## Verified changes

17 tests pass, including nonfinite settings, conversion overflow/underflow, malformed model metadata, strict reset fields, depth boundaries, extreme motion and equivalence to the old full-residual calculation at all supported scales. CLI demos ran at 2x, 3x and 4x. The supplied 3x sequence ran successfully and produces a manifest. Training data and model size are unchanged.

At 3x, mean held-out gain remains +0.397834 dB over bilinear, with +0.220059 dB over bicubic and +0.152658 dB over Lanczos on six procedural scenes. These are synthetic-only PSNR results.

A direct same-runtime old/new comparison at 256x256 to 768x768 measured traced peak allocations of 21.00 versus 15.01 MiB (approximately 29% lower), and CPU median inference of 64.39 versus 56.41 ms. Fifteen timing samples followed warmup; memory was measured separately with tracemalloc. Traced allocations are not process RSS, and timing variability prevents a universal speedup claim. Measurements are in demo/inference_comparison.json.

## Temporal findings

The harness evaluates three unseen seeds, each with clean and noisy translating input, for sixteen frames per case (96 frames total). High-resolution truth is known. It reports mean RGB MSE and motion-aligned changes in reconstruction error; the latter is a custom diagnostic, not a standard perceptual score. It uses spatial bilinear and temporal bilinear reconstruction, without the learned model, to isolate history behavior. Border crops exclude the wrap seam.

History reduces mean reconstruction MSE modestly in all six cases and substantially reduces motion-aligned error fluctuation in the noisy cases. Clean translation cases already have essentially zero spatial error fluctuation; the temporal path introduces roughly 3.4e-6 to 4.7e-6 diagnostic error. This exposes history/clipping/boundary sensitivity and is not a passed general temporal-fidelity gate.

The harness does not validate silhouettes, moving-camera depth, particles, jitter, exposure, HUD text or real gameplay. Nearest metadata sampling has a boundary-preservation regression test but no demonstrated visual-quality advantage over the old temporal policy. The temporal change needs representative captures before being treated as a quality improvement.

## Remaining work

1. Add nonwrapping captures with foreground/background surfaces, disocclusion masks and full motion-ground-truth evaluation; isolate the clean-translation fluctuation.
2. Test representative licensed/owned real-image holdouts and inspect ringing, fine text and foliage.
3. Verify installation on clean Python 3.11+ and pin an environment for reproducible releases.
4. Measure actual peak process memory and larger-resolution latency before selecting a GPU backend.
5. Add camera-aware depth reprojection, jitter/exposure and engine buffer integration only after the evaluation contract is established.

No GPU, HDR, arbitrary-camera, neural, game-injection or general hardware support claim is added by this release. The original roadmap and calendar are retained as planning history, not evidence of currently scheduled automation.
