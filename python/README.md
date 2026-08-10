# `python/`

The training pipeline and Python-side reference implementation. This is where the model was
designed, trained and tuned, and where the onset-detection logic was developed and evaluated
before being ported to C.

| File | Overview |
|---|---|
| `config.py` | Central configuration: HCQT parameters, the 11 instrument labels, dataset paths, and all training hyperparameters (batch size, learning rate, weight decay, early-stopping/LR-schedule settings). |
| `models_config.py` | Registry of the five model architectures that were compared (`slim_vgg`, `wider_vgg`, `slim_resnet`, `wider_resnet`, `vgg_multihead`). `config.py` selects which one is active. |
| `model.py` | Defines the CNN architectures: `VGGBaseModel` and `ResNetBase` (the two candidates compared in the report), and `VGGMultiHead`, the deployed multi-head architecture. |
| `hcqt.py` | Computes the Harmonic Constant-Q Transform using `librosa`, and slices it into fixed-size windows for training/inference. This is the Python reference feature extractor; the deployed version is `c/src/cqt.c`. |
| `hcqt_preprocessing.py` | Builds the training/validation/test `InstrumentDataset`s: pre-computes and HDF5-caches HCQTs and frame-level labels, applies label smoothing, and hooks in data augmentation. This is the version actually imported by `train_instrument_model.py` and `bayesian_optimisation.py`. |
| `print_functions.py` | Formatting and logging helpers used by the dataset-building and training scripts (dataset statistics, cache summaries). |
| `data_processing.py` | Dataset statistics and caching, per-instrument class-weight calculation for the loss function, and `find_optimal_thresholds`, the F1-optimal decision threshold search saved into the trained checkpoint. |
| `augmentations.py` | Thin wrapper around `torchaudio`'s frequency and time masking for training-time data augmentation. |
| `loss_function.py` | A from-scratch, numerically-stable binary cross-entropy-with-logits implementation, verified against `torch.nn.BCEWithLogitsLoss` on a fixed test vector. |
| `Optimizer.py` | A from-scratch AdamW optimiser (decoupled weight decay, bias-corrected moment estimates), implemented in plain NumPy as a from-first-principles reference. |
| `early_stopping.py` | Patience/delta-based early stopping, with checkpoint saving (model, optimiser, scaler, thresholds) whenever validation loss improves. |
| `train_instrument_model.py` | The main training loop and evaluation harness (per-class/macro precision, recall, F1, frame accuracy, Hamming loss). |
| `bayesian_optimisation.py` | Optuna-based hyperparameter search (learning rate, weight decay, per-head dropout) that produced the final training configuration in `config.py`. |
| `activity_single.py` | `ActivityDetector`, which loads the trained model and runs the full inference pipeline (HCQT, windowing, batched forward pass, aggregation) on a single audio file. Everything downstream builds on this class. |
| `activity_gradient.py` | `FluxPeriodFinder`, which detects periods of sharply rising instrument probability, referred to as flux, in the prediction timeline. One half of the onset-detection logic. |
| `activity_period.py` | `ActivityPeriodFinder`, which thresholds and de-noises the probability timeline into sustained activity periods. The other half of the onset-detection logic. |
| `activity_eval_test.py` | Evaluation harness for frame-level instrument activity detection, as opposed to onset timing. Computes precision, recall and F1 against ground-truth labels. |
| `onset_detector.py` | `InstrumentAnalysisOrchestrator`, which combines activity periods and flux periods into confirmed onsets, plus the final per-instrument tuned detection parameters. |
| `onset_eval.py` | Scores predicted onsets against ground-truth onset annotations (precision, recall, F1 within a time tolerance). This produces the onset-detection results reported in the write-up. |
| `onset_param_tune.py` | Parallelised grid search over onset-detection parameters (flux threshold, minimum activity duration, significance score, and so on) per instrument. |
| `alignment.py` | Cross-correlates original vs. re-recorded (speaker-to-mic) audio to find and correct playback/recording latency, then trims to match. Used to build the re-recorded training set. |
| `record_data.py` | Companion script that plays back original audio through a speaker and records it via microphone, producing the raw re-recorded audio that `alignment.py` then aligns. |
