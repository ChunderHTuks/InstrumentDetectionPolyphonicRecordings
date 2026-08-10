# `c/`

The deployed inference engine: everything that actually runs on the Jetson Nano. There is no
inference framework (no ONNX runtime, TensorRT or libtorch) - the forward pass, the feature
extraction and the onset-detection logic are all hand-written C, optimised with OpenMP, ARM NEON
SIMD intrinsics and cache-aware loop tiling.

| File | Overview |
|---|---|
| `include/model_params.h` | Auto-generated (by the Python-side export script) static description of every layer's shape. Lets the rest of the C code stay shape-agnostic instead of hardcoding dimensions. |
| `include/cpu_layers.h`, `src/cpu_layers.c` | The core math: tiled, OpenMP-parallel, NEON-vectorised 2D convolution, leaky ReLU, sigmoid, and instance normalisation. This is the performance-critical file, with an 8x16 cache-aware tiling scheme sized for the Jetson Nano's L1/L2 cache. |
| `include/cqt.h`, `src/cqt.c` | A from-scratch Constant-Q Transform built on FFTW. Generates and caches per-bin logarithmically-spaced kernels, then computes each frame via windowed FFT and a frequency-domain inner product. |
| `include/weight_loader.h`, `src/weight_loader.c` | Loads the exported binary weight files into memory: the BatchNorm-folded backbone/head weights, the un-folded classifier weights, and the instance-normalisation parameters. |
| `include/onset_detector.h`, `src/onset_detector.c` | C port of the Python onset-detection logic (probability flux plus activity periods), operating on the aggregated probability timeline produced by `main_inference.c`. |
| `include/sound_acquisition.h`, `src/sound_acquisition.c` | Live microphone capture via ALSA (44.1 kHz, mono, 16-bit), converted to normalised float32. |
| `include/utils.h`, `src/utils.c` | Small shared helpers: binary file loading, timing and formatting utilities. |
| `include/dr_wav.h` | Third-party single-header WAV file library ([`mackron/dr_libs`](https://github.com/mackron/dr_libs)), used for reading and writing `.wav` files. Not written by me, included for convenience. |
| `src/main_inference.c` | The orchestrator. Loads or records audio, pads it, computes the full-clip HCQT, loads model weights, slides a window over the HCQT running the CNN forward pass on each, aggregates overlapping predictions, runs onset detection, then prints results and timing statistics. |
| `src/main_verification.c` | Standalone verification tool: runs a Python-generated input tensor through the full C forward pass, saving every intermediate layer's output to `verification_data/` for direct numerical comparison against the equivalent PyTorch activations. This is what produced the layer-by-layer error table in the report. |
| `src/main_dataprep_verifier.c` | Verifies the C HCQT/audio-padding pipeline against Python by processing the same `.wav` file and saving intermediate tensors for comparison. |
| `src/main_instancenorm_verifier.c` | Verifies the from-scratch instance normalisation kernel against Python, using a C-generated HCQT as input (depends on `main_dataprep_verifier` having been run first). |
| `src/main_aggregation_verifier.c` | Verifies the sliding-window prediction aggregation and thresholded display logic against Python's raw per-window predictions. |

Running `make` produces the main `instrument_detector` binary. `make verifier`, `make dataprep`,
`make instancenorm` and `make agg` build the four verification tools above individually.

**Note on weights.** The exported, BatchNorm-folded binary weight files that `weight_loader.c`
expects under `assets/c_weights/` are not included in this repository, only the code is. To
regenerate them, export a trained checkpoint's parameters to raw float32 binaries, fusing every
`Conv2d -> BatchNorm2d` pair into a single folded convolution weight/bias. `weight_loader.c`'s
`load_folded_conv_block()` expects one `<name>_folded_weight.bin`/`<name>_folded_bias.bin` pair
per convolution; see `include/model_params.h` for the exact layer names and shapes.
