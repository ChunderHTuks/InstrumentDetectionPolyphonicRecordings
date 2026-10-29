"""
Central settings for the whole project: HCQT parameters, dataset paths, instrument labels
and all training hyperparameters. Every other module imports from here.
"""
import os

# Key into MODEL_BLUEPRINTS in models_config.py
ACTIVE_MODEL_NAME = 'vgg_multihead'

# HCQT CONFIG

SAMPLE_RATE = 44100
# Hop sizes are in samples. 512 samples at 44.1 kHz is one frame every 11.61 ms
HOP_LENGTH = 512
# 84 bins at 12 per octave covers 7 octaves of semitones
N_BINS = 84
BINS_PER_OCTAVE = 12
# Number of stacked CQTs in the HCQT (the fundamental plus 4 more harmonics)
N_HARMONICS = 5
CQT_HOP = 512

# TRAINING CONFIG

# Dataset paths are relative to the folder the script is run from
FINAL_DATASET_DIR = './Dataset'

TRAIN_AUDIO_DIR = os.path.join(FINAL_DATASET_DIR, 'Training', 'train_data')
TRAIN_LABEL_DIR = os.path.join(FINAL_DATASET_DIR, 'Training', 'train_labels')

VAL_AUDIO_DIR = os.path.join(FINAL_DATASET_DIR, 'Validation', 'validation_data')
VAL_LABEL_DIR = os.path.join(FINAL_DATASET_DIR, 'Validation', 'validation_labels')

TEST_AUDIO_DIR = os.path.join(FINAL_DATASET_DIR, 'Testing', 'test_data')
TEST_LABEL_DIR = os.path.join(FINAL_DATASET_DIR, 'Testing', 'test_labels')

INSTRUMENTS_LABELS = ['Piano', 'Violin', 'Cello', 'Viola', 'Contrabass', 'Clarinet', 'Bassoon','Oboe', 'Flute', 'French Horn', 'Trumpet']

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0021963789036605324
NUM_INSTRUMENTS = 11
# Maximum L2 norm of the gradients before they are clipped
GRAD_CLIP_NORM = 1
# Length of one input patch, and how far the patch window moves each step
FRAME_SIZE_MS = 500
HOP_SIZE_MS = 500

# Cached HCQTs are stored under CACHE_DIR in a folder named after the HCQT settings
USE_CACHE = True
CACHE_DIR = './cache'
CACHE_NAME = f"{N_HARMONICS}_{SAMPLE_RATE}_{HOP_LENGTH}_{N_BINS}_{BINS_PER_OCTAVE}"

# Softens the 0 and 1 labels by this amount
LABEL_SMOOTHING = 0.02
RANDOM_STATE = 42

OPTIM_WEIGHT_DECAY = 2.1503642094991787e-05
OPTIM_EPS = 1e-8

# Early stopping: epochs to wait for an improvement, and the smallest change that counts
ES_PATIENCE = 6
ES_DELTA = 0.0001
ES_METRIC = 'Loss'

# Epochs over which the learning rate is ramped up at the start of training
WARMUP_EPOCHS = 5

# Reduce-on-plateau learning rate scheduler settings
LRS_MODE = 'min'
LRS_FACTOR = 0.1
LRS_PATIENCE = 3

# DataLoader settings
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
PERSISTENT_WORKERS = True

CQT_FRAME_DURATION_MS = (CQT_HOP / SAMPLE_RATE) * 1000
# Onset timing tolerance used when scoring predicted onsets against annotations
TEMPORAL_TOLERANCE_MS = 100

# Fraction of the frequency bins and time steps that the masking augmentation can cover.
# 0.0 means the masks have zero width, so no masking is applied
FREQ_MASK_PARAM = 0.0
TIME_MASK_PARAM = 0.0