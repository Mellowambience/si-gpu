# Architecture and evidence boundary

Current spatial path: display RGB -> bilinear output -> learned polyphase residual correction -> clipping to [0,1]. `LearnedUpscaler.fit` estimates ten coefficients per output phase shared across RGB channels. The estimator sees twelve synthetic training scenes, then evaluation uses six unseen seeds. This is a trained **linear** filter, not a deep neural model.

Current temporal path: spatial result + low-resolution depth/motion/reactive data -> output-resolution motion reprojection -> reject invalid/out-of-bounds/depth-mismatched history -> clip valid history to a 3x3 current neighborhood -> blend -> retain output and depth. It has tests for functional behavior but no demonstrated improvement in moving gameplay.

Planned neural path: compact residual CNN -> subpixel reconstruction -> validated temporal fusion -> GPU backend -> engine adapter. Choose a permissive training dataset and preserve dataset/checkpoint provenance. Do not run untrained neural weights as an improvement demo. Do not synthesize replacement textures or faces for the first product: prioritize source fidelity.

Backend choice remains open. Start with one supported backend on the 2080 Ti, then validate a separate hardware/backend pair. An NVIDIA-only first development backend is a stepping stone, not evidence of vendor neutrality. Portable GPU compute must be profiled rather than presumed fast.

Do not place long-lived image allocations, GPU copies or synchronization on the CPU hot path without measuring them. The reference NumPy implementation is deliberately easy to inspect and memory-heavy; it is not the realtime architecture. Add GPU timestamps, warmups, repeated samples, memory budgets and actual end-to-end input latency measurement in M5/M6.

Frame generation is deferred. Presenting more frames is not the same as increasing rendered simulation FPS or improving input response. Never combine generated and rendered FPS in a misleading benchmark.

## Required evaluation

Compare against native high-res ground truth, bilinear, bicubic and a relevant engine temporal baseline. Use scenes excluded from training, consistent jitter/exposure and equal output resolution. Record PSNR, temporal residual error using ground-truth motion, human artifact review and p50/p95 GPU time. PSNR alone is not a quality verdict. Report foliage shimmer, motion trails, disocclusions, UI readability, thin lines and particles separately.

## External primary references

AMD's temporal integration reference documents motion/depth/jitter requirements: https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-temporal/

FSR 2's permissive source is a potential later baseline, not bundled here: https://gpuopen.com/fidelityfx-superresolution-2/

PyTorch PixelShuffle documents a neural subpixel output operation for M3: https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.pixelshuffle.PixelShuffle.html


## v0.1.1 implementation update

Inference accumulates residuals per output phase rather than materializing all phase features. Depth/motion enlargement and prior depth lookup use nearest sampling to preserve surface values; this is not full nearest-depth dilation. See REFINEMENT.md for research and measured limitations.
