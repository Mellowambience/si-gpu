# Frame contract v0.1

`python -m siware sequence` consumes lexically ordered `*.npz` files. Use padded frame numbers: `00000.npz`, `00001.npz`, etc. Always store arrays without object types; loader disables pickle.

| Field | Shape/type | Meaning |
|---|---|---|
| color | H x W x 3, float32 in [0,1] | Display RGB; not HDR. Minimum 2x2. |
| depth | H x W, positive finite float32-representable float | Linear view depth with identical units and comparable camera space between frames. |
| motion | H x W x 2, finite float | Current pixel to previous pixel displacement, low-resolution pixels. x right, y down. A feature that moved right by 2 pixels has motion x=-2. |
| reactive | Optional H x W in [0,1] | 1 disables history for particles/transparency/UI or unstable regions. |
| reset | Optional boolean scalar | True on camera cuts, projection/exposure changes, or discontinuities. |

Only **unjittered** frames are supported today. Scale is 2, 3 or 4. The output has H*scale by W*scale RGB values. Motion displacement is multiplied by scale for output-resolution history sampling. Dimension changes discard prior history. Depth rejection uses relative threshold 0.02 by default; it is a heuristic, not production occlusion handling.

Future protocol: render/display dimensions, frame ID, timestamp, camera matrices, jitter offsets, exposure, color space, depth convention, reset reasons. Add projection-aware depth reprojection before accepting arbitrary camera movement.

Keep HUD layers separate and composite after upscaling in the eventual engine integration. Current reactive masks can only suppress history; they cannot restore low-resolution text detail.

In v0.1.1, depth and motion are converted to float32 and validated after conversion. Positive depth that underflows to zero is rejected. Depth/motion enlargement and history-depth lookup use nearest sampling; color history still uses bilinear sampling. `reset` must be an actual boolean scalar, not an integer or string. Temporal settings must be finite; depth threshold lies in [0,1].
