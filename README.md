# Instrument Detection in Polyphonic Recordings

An embedded system that listens to a live classical chamber-music performance (duets through
quintets) and detects, in real time, which of 11 instruments are playing and when each one enters
the mix. Built as my final-year Computer Engineering capstone project (University of Pretoria,
EPR402, 2025).

Everything in this repository is implemented from first principles: the audio feature extraction,
the CNN, the training loop, the optimiser, the loss function and the onset-detection logic. The neural network is trained in PyTorch, but the
inference engine that runs on the target hardware is a hand-written C implementation that
re-derives the forward pass, convolution and Constant-Q Transform from scratch.

**Full write-up:** [`docs/22659511_ChunderH_FINALREPORT.pdf`](docs/22659511_ChunderH_FINALREPORT.pdf).
It covers the literature review, the architecture trade-off analysis, the verification
methodology and the full set of results. This README is a shorter version aimed at a code
reviewer.

## What it does

Given a polyphonic audio stream (live microphone input or a `.wav` file), the system reports, for
each of piano, violin, viola, cello, contrabass, clarinet, bassoon, oboe, flute, French horn and
trumpet, whether it is present, its per-frame confidence over time, and the timestamp of its first
onset in the mix.

| Metric (100 ms onset tolerance) | Target | Achieved |
|---|---|---|
| Macro F1 | ≥ 0.75 | **0.743** |
| Macro precision | ≥ 0.77 | **0.741** |
| Macro recall | ≥ 0.73 | **0.768** |
| Real-time processing factor (Jetson Nano) | ≤ 2.0x clip duration | **1.81x** average |

The dominant failure mode is false positives rather than missed detections. This is traced back to
heavy instrument co-occurrence in the training data (for example, viola co-occurs with
violin/cello 78-86% of the time in the compiled dataset). See §4.2/§5 of the report for the full
discussion.

## How it works

![System block diagram](docs/figures/system_block_diagram.png)

**Feature representation.** The HCQT stacks 5 CQTs at `f_min, 2f_min, 3f_min, 4f_min, 5f_min`
(`f_min` = C1, 32.7 Hz), so that each frequency bin's harmonic series lines up across channels.
This makes it much easier for the CNN's early layers to learn timbral features than a plain
spectrogram would. On the training side this is `librosa.cqt` (`python/hcqt.py`); the deployed
version is a from-scratch, FFTW-based implementation (`c/src/cqt.c`). Report §3.5 covers why HCQT
was chosen over Mel-spectrograms and MFCCs for this task.

**Model.** A multi-head VGG-style CNN (`VGGMultiHead` in `python/model.py`). A shared 6-layer
convolutional backbone downsamples along the frequency axis only, so time resolution is preserved
throughout, followed by four independent classification heads, one per instrument family (piano,
strings, woodwinds, brass) rather than one shared 11-way head.

![Multi-head architecture](docs/figures/multihead.png)

Report §3.6 documents the architecture search (VGG vs. ResNet) that led here. Class imbalance
(violin and piano appear roughly 15x more often than contrabass and trumpet in the compiled
dataset) is handled through `BCEWithLogitsLoss` positive class weighting and per-instrument
F1-optimal decision thresholds, rather than by discarding data.

**Onset detection.** Fully rule-based, running on the CNN's output probability timeline with no
further learning involved. An instrument is flagged as "active" once its probability crosses a
tuned threshold for long enough (`python/activity_period.py`), and an "onset" is confirmed where
an activity period overlaps a sharp rise in probability, referred to as "flux"
(`python/activity_gradient.py`). Per-instrument thresholds and flux/activity parameters were tuned
by exhaustive grid search against a held-out validation set (`python/onset_param_tune.py`).

**Deployment.** The trained PyTorch model's weights are exported and BatchNorm-folded: every
`Conv2d -> BatchNorm2d` pair is algebraically fused into a single convolution, so the C engine only
ever needs to run a plain conv+bias (see `docs/22659511_ChunderH_FINALREPORT.pdf` §3.6.5 for the
derivation and a layer-by-layer numerical verification against PyTorch, which typically agrees to
around 1e-6 absolute error). The result is exported into flat binary files consumed by
`c/src/weight_loader.c`. This repository ships the code, not the trained weights - see
`c/README.md` for what's needed to regenerate them. The C inference engine (`c/`) loads them
directly, with no ONNX runtime, TensorRT or libtorch involved, and runs the full HCQT -> CNN ->
onset pipeline on a Jetson Nano's ARM Cortex-A57 CPU. It uses OpenMP for layer-level parallelism,
ARM NEON SIMD intrinsics in the inner convolution loop, and cache-aware loop tiling (8x16 tiles
sized to fit the Nano's L1/L2 cache).

## Repository structure

```
InstrumentDetectionPolyphonicRecordings/
├── python/     Training pipeline: dataset loading, model definition, loss/optimiser (both with
│               from-scratch verification against PyTorch equivalents), HPO, and the reference
│               (Python-side) onset-detection implementation used during development/evaluation.
│               See python/README.md for a file-by-file breakdown.
├── c/          Deployed inference engine: HCQT, convolution/activation kernels, weight loading,
│               onset detection, audio capture - everything that runs on-device.
│               See c/README.md for a file-by-file breakdown.
└── docs/       Full final report plus the figures embedded in this README.
                See docs/README.md.
```

Each subfolder has its own `README.md` with a short summary of every file in it. Start there if
you want to jump straight to a specific piece, such as the CQT implementation, the onset detector,
or the from-scratch optimiser.

## Verification methodology

Every from-scratch component was checked against a reference implementation before being trusted.
For the CQT, the custom FFTW-based C implementation was compared bin-by-bin against `librosa.cqt`
on the same audio. The difference is essentially at the noise floor, with no structural pattern:

![CQT verification: custom C implementation vs. librosa, per-bin difference](docs/figures/cqt_difference.png)

- **Forward pass:** the full folded C model against the live PyTorch model, layer by layer, on
  identical random input. Mean absolute error is on the order of 1e-7, max around 1e-6, which is
  consistent with single-precision floating point rounding rather than a genuine mismatch.
- **Loss function:** the from-scratch, numerically-stable BCE-with-logits implementation against
  `torch.nn.BCEWithLogitsLoss` on a fixed test vector. Outputs match to 8 decimal places.
- **Optimiser:** the from-scratch AdamW update rule, implemented in NumPy.

## Known limitations

- Performance is bottlenecked by false positives from instrument co-occurrence bias in the
  training data, particularly within the string and woodwind families, rather than by an inability
  to detect onsets. See the report's discussion (§5/§6.2) for the full analysis and suggested
  fixes.
- There is still a measurable domain gap between the training data, even after re-recording
  through a real speaker/mic chain, and arbitrary real-world recording conditions. Report §3.10.2
  quantifies this using Pearson correlation and cosine similarity of HCQTs across recording
  domains.
- This repository is a cleaned-up snapshot of a research/capstone codebase, not a maintained
  library. See `docs/` for the full list of design trade-offs and suggested future work.

## Acknowledgements

Supervised by Prof. J.P. Jacobs, Department of Electrical, Electronic and Computer Engineering,
University of Pretoria. WAV file I/O in the C engine uses the single-header
[`dr_wav`](https://github.com/mackron/dr_libs) library by David Reid.
