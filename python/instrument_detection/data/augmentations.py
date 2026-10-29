"""
Training-time augmentation: frequency masking and time masking on the HCQT patches.
"""
import torch
import torchaudio.transforms as T
from instrument_detection.config import N_BINS, FREQ_MASK_PARAM, TIME_MASK_PARAM

# Number of CQT frames in one 500 ms patch
N_TIME_STEPS = 44

def get_train_augmentations():
    """
    Builds the frequency masking and time masking pipeline.
    Scales the mask widths from the fractions in config.py into bins and frames.
    """
    augment_pipeline = torch.nn.Sequential(
        T.FrequencyMasking(
            freq_mask_param=int(N_BINS * FREQ_MASK_PARAM)
        ),

        T.TimeMasking(
            time_mask_param=int(N_TIME_STEPS * TIME_MASK_PARAM)
        )
    )

    return augment_pipeline