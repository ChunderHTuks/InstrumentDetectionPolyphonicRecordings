# `python/`

The training pipeline and Python-side reference implementation. This is where the model was
designed, trained and tuned, and where the onset-detection logic was developed and evaluated
before being ported to C.

All code lives in the `instrument_detection` package, so each file below is listed under its sub-folder of `python/instrument_detection/`. Run scripts from the `python/` folder as modules, for example:

```
python -m instrument_detection.training.train_instrument_model
python -m instrument_detection.onset.onset_eval
```

Dataset, cache and model paths in `config.py` are relative to the folder you run from, so keep running from `python/`.

## Setup

The code was written against these packages: torch, torchaudio, librosa, numpy, scipy, pandas,
h5py, optuna, scikit-learn, scikit-multilearn, soundfile, sounddevice, matplotlib and tqdm.
The dataset is not included. Put it in `Dataset/` (inside the folder you run from) with the
`Training/`, `Validation/` and `Testing/` sub-folders and audio/label split that `config.py`
describes. Trained checkpoints are saved to and read from `models/`, and cached HCQTs go to `cache/`.

## Package root

| File | Overview |
|---|---|
| `config.py` | Central configuration: HCQT parameters, the 11 instrument labels, dataset paths, and all training hyperparameters (batch size, learning rate, weight decay, early-stopping/LR-schedule settings). |
| `print_functions.py` | Formatting and logging helpers used by the dataset-building and training scripts (dataset statistics, cache summaries). |

## `models/`

Network architectures.

| File | Overview |
|---|---|
| `models_config.py` | Registry of the five model architectures that were compared (`slim_vgg`, `wider_vgg`, `slim_resnet`, `wider_resnet`, `vgg_multihead`). `config.py` selects which one is active. |
| `model.py` | Defines the CNN architectures: `VGGBaseModel` and `ResNetBase` (the two candidates compared in the report), and `VGGMultiHead`, the deployed multi-head architecture. |

## `data/`

Feature extraction, dataset building and augmentation.

| File | Overview |
|---|---|
| `hcqt.py` | Computes the Harmonic Constant-Q Transform using `librosa`, and slices it into fixed-size windows for training/inference. This is the Python reference feature extractor; the deployed version is `c/src/cqt.c`. |
| `hcqt_preprocessing.py` | Builds the training/validation/test `InstrumentDataset`s: pre-computes and HDF5-caches HCQTs and frame-level labels, applies label smoothing, and hooks in data augmentation. This is the version actually imported by `train_instrument_model.py` and `bayesian_optimisation.py`. |
| `data_processing.py` | Dataset statistics and caching, per-instrument class-weight calculation for the loss function, and `find_optimal_thresholds`, the F1-optimal decision threshold search saved into the trained checkpoint. |
| `augmentations.py` | Thin wrapper around `torchaudio`'s frequency and time masking for training-time data augmentation. |

## `training/`

Training, hyperparameter search and the from-scratch loss and optimiser.

| File | Overview |
|---|---|
| `train_instrument_model.py` | The main training loop and evaluation harness (per-class/macro precision, recall, F1, frame accuracy, Hamming loss). |
| `bayesian_optimisation.py` | Optuna-based hyperparameter search (learning rate, weight decay, per-head dropout) that produced the final training configuration in `config.py`. |
| `early_stopping.py` | Patience/delta-based early stopping, with checkpoint saving (model, optimiser, scaler, thresholds) whenever validation loss improves. |
| `loss_function.py` | A from-scratch, numerically-stable binary cross-entropy-with-logits implementation, verified against `torch.nn.BCEWithLogitsLoss` on a fixed test vector. |
| `adamw_optimiser.py` | A from-scratch AdamW optimiser (decoupled weight decay, bias-corrected moment estimates), implemented in plain NumPy as a from-first-principles reference. |
| `adamw_optimiser_test.py` | Checks the from-scratch AdamW against `torch.optim.AdamW` on the 3D Rosenbrock function (start [-1, 2, -0.5], learning rate 0.01, weight decay 0.01, 1000 steps) and saves four comparison plots as PDFs in the folder you run it from. The maximum relative error in the loss is about 6.9e-16. |

## `activity/`

Frame-level instrument activity detection.

| File | Overview |
|---|---|
| `activity_single.py` | `ActivityDetector`, which loads the trained model and runs the full inference pipeline (HCQT, windowing, batched forward pass, aggregation) on a single audio file. Everything downstream builds on this class. |
| `activity_eval.py` | Evaluation harness for frame-level instrument activity detection, as opposed to onset timing. Computes precision, recall and F1 against ground-truth labels. |

## `onset/`

Rule-based onset detection on top of the activity probabilities.

| File | Overview |
|---|---|
| `activity_period.py` | `ActivityPeriodFinder`, which thresholds and de-noises the probability timeline into sustained activity periods. The other half of the onset-detection logic. |
| `activity_gradient.py` | `FluxPeriodFinder`, which detects periods of sharply rising instrument probability, referred to as flux, in the prediction timeline. One half of the onset-detection logic. |
| `onset_detector.py` | `InstrumentAnalysisOrchestrator`, which combines activity periods and flux periods into confirmed onsets, plus the final per-instrument tuned detection parameters. |
| `onset_eval.py` | Scores predicted onsets against ground-truth onset annotations (precision, recall, F1 within a time tolerance). This produces the onset-detection results reported in the write-up. |
| `onset_param_tune.py` | Parallelised grid search over onset-detection parameters (flux threshold, minimum activity duration, significance score, and so on) per instrument. |

## `export/`

Turns a trained checkpoint into the files the C engine needs.

| File | Overview |
|---|---|
| `export_folded_weights.py` | Loads a trained checkpoint, fuses every `Conv2d -> BatchNorm2d` pair into one convolution, and writes the folded weights, the input-norm parameters and the classifier weights as raw float32 files to `../c/assets/c_weights/`. Set `MODEL_PATH` at the top of the file to your checkpoint. |
| `export_model_architecture.py` | Reads the layer shapes from the model and writes the C header `../c/include/model_params.h`, so the C code does not hardcode any dimensions. Run it again whenever the architecture changes. |

## `recording/`

Speaker-to-microphone re-recording tools.

| File | Overview |
|---|---|
| `record_data.py` | Companion script that plays back original audio through a speaker and records it via microphone, producing the raw re-recorded audio that `alignment.py` then aligns. |
| `alignment.py` | Cross-correlates original vs. re-recorded (speaker-to-mic) audio to find and correct playback/recording latency, then trims to match. Used to build the re-recorded training set. |
