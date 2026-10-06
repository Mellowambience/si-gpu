# SI-WARE v0.1.0 analysis

SI-WARE is a working, compact offline reconstruction research prototype. Its implementation and documentation are consistent about its limits. It establishes a useful experimental baseline, but provides no evidence yet for real-time gameplay, GPU acceleration, neural reconstruction, or multiple-device support.

## Architecture

- Spatial reconstruction: channel-wise bilinear enlargement followed by a trained 3x3 linear residual filter. There are ten coefficients per output phase, shared across RGB: 90 coefficients at 3x.
- Training: ridge regression on twelve procedural scenes. Evaluation uses six different seeds from the same generator. This is a sensible reproducibility check, but does not establish generalization to gameplay.
- Temporal reconstruction: current-to-previous motion reprojection, relative-depth rejection, 3x3 neighborhood clipping, reactive-mask suppression, and history resets.
- Interfaces: Python CLI commands for demo generation, single-image enlargement, and NPZ frame sequences. Dependencies are NumPy and Pillow. There is no GPU backend, engine adapter, or installable package metadata.

## Verification performed

The ZIP was extracted into workspace scratch storage; the original archive and source were not edited.

On the available Windows Python runtime (Python 3.10.1, NumPy 2.2.6, Pillow 12.1.0):

- All ten bundled unit tests passed.
- The demo retrained successfully and reproduced the reported mean PSNR improvement of +0.397834 dB at 64x64 to 192x192.
- All six evaluation scenes improved against bilinear; individual gains were approximately +0.361 to +0.421 dB.
- Median inference times per scene ranged from 2.82 to 3.64 ms on these tiny inputs, with five timed samples per scene. These measurements do not establish large-image, GPU, or end-to-end performance.
- The sequence CLI successfully reconstructed all eight supplied frames.
- The single-image CLI successfully used the supplied model.

The README requests Python 3.11+, while the available runtime was 3.10.1. Successful execution here is additional compatibility evidence, not verification of the recommended clean-install environment. The bundled environment.json describes an earlier Linux run, not this Windows verification.

## Findings, ordered by practical importance

### 1. Temporal quality is still unproven

The tests verify mechanics such as reset behavior, motion sign, and rejection. They do not compare reconstructed motion against high-resolution moving ground truth or measure ghosting, shimmer, and disocclusion errors. The sample sequence has no supplied high-resolution target. Treat temporal processing as functional reference code until an evaluation harness demonstrates benefit over the spatial path.

### 2. Interpolation can mix foreground and background metadata

core.py resizes both depth and motion using bilinear interpolation. At silhouette boundaries, interpolated depth can represent neither surface, and interpolated motion can combine unrelated movements. History acceptance can therefore be unreliable near edges. This is an inferred algorithmic risk from the implementation, not a measured artifact in the supplied demo. Evaluate edge-aware metadata selection with controlled moving silhouettes before engine integration.

### 3. Nonfinite configuration values pass validation

Confirmed by runtime probes: TemporalReconstructor accepts depth_threshold=NaN, and LearnedUpscaler.fit accepts ridge=NaN and produces nonfinite weights. Comparisons such as threshold < 0 and ridge <= 0 do not reject NaN. Validate finiteness explicitly and test malformed configuration values.

### 4. Depth validation precedes a potentially overflowing conversion

Confirmed by runtime probe: positive, finite float64 depth values of 1e100 pass process validation, then overflow when resize converts them to float32. Stored history depth becomes infinite. Such magnitudes are unrealistic for ordinary scene depth, but the documented positive-finite contract does not exclude them. Validate after conversion or reject values outside the supported representation.

### 5. CPU allocations prevent performance extrapolation

Spatial inference materializes H x W x 3 x 10 features and H x W x 3 x scale² residuals. For 1280x720 input at 3x, those arrays alone total approximately 200 MiB, plus an approximately 95 MiB output and temporary arrays. Temporal sampling introduces output-sized arrays, float64 interpolation intermediates, neighborhood reductions, and copies. These are estimates from shapes and dtypes, not a measured peak. Profile time and peak memory at gradually larger sizes before defining hardware requirements.

### 6. Quality and performance coverage are narrow

The demo compares only bilinear and the learned filter. There is no bicubic/Lanczos comparison, perceptual evaluation, real-image holdout, temporal metric, p95 timing, or large-resolution benchmark. The existing training roundtrip test uses a bilinear target, so it verifies fitting/persistence but not useful learned improvement. CLI verification here covers the supplied 3x configuration; 2x/4x demo behavior and malformed archives were not exhaustively tested.

### 7. Packaging and provenance need a release pass

There is no pyproject.toml or dependency lock. Regenerated demos do not write environment.json, leaving environment provenance manual. Sequence outputs are renumbered, and no manifest retains input-to-output mapping. The roadmap also states reminders were scheduled; an archive cannot verify their current existence. Calendar import and automation state were not inspected or changed.

## Recommended next work

1. Fix finite-value validation and conversion overflow, with focused regression tests.
2. Build paired moving ground truth and compare spatial versus temporal reconstruction across silhouettes, thin lines, disocclusions, and reactive content.
3. Add bicubic/Lanczos and real-image held-out comparisons before increasing model complexity.
4. Record runtime provenance automatically and measure larger-input peak memory and latency.
5. Use the resulting quality evidence to choose whether neural training or temporal corrections should come next; GPU and engine integration remain later milestones.

The strongest part of this release is its small, understandable implementation and reproducible evidence. Its main gap is evaluation breadth. Expanding that evidence is more valuable now than making broader graphics or hardware claims.
