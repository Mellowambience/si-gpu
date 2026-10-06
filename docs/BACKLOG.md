# Backlog

| ID | Priority | Task | Dependency | Status |
|---|---|---|---|---|
| LAB-01 | P0 | CPU learned residual and baseline | None | Built, host tested |
| LAB-02 | P0 | Motion/depth/reactive history processing | LAB-01 | Built, functional tests pass |
| LAB-03 | P0 | Windows local verification | LAB-01 | Pending on target PC |
| DATA-01 | P0 | Capture contract with camera/jitter/exposure | LAB-03 | Planned |
| DATA-02 | P0 | Scene-separated native/low-res dataset | DATA-01 | Planned |
| EVAL-01 | P0 | Moving ground-truth metric harness | DATA-02 | Planned |
| MODEL-01 | P1 | Compact neural training + held-out model card | DATA-02 | Planned |
| TEMP-01 | P1 | Jitter, exposure and moving-camera depth support | EVAL-01 | Planned |
| GPU-01 | P1 | GPU execution and timestamp/memory benchmarks | MODEL-01, TEMP-01 | Planned |
| ENGINE-01 | P1 | Controlled engine capture + reconstruction adapter | GPU-01 | Planned |
| PORT-01 | P1 | Second actual device/backend validation | GPU-01 | Planned |
| RELEASE-01 | P1 | Experimental SDK + clean-machine instructions | ENGINE-01 or offline-only fallback | Planned |
| FG-01 | P2 | Optional frame-generation research | Stable reconstruction release | Deferred |

Select one active engineering task at a time. Definition of done: working output, appropriate validation, retained measurements and explicit remaining limitations.
