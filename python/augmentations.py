import torch
import torchaudio.transforms as T
from config import N_BINS, FREQ_MASK_PARAM, TIME_MASK_PARAM

N_TIME_STEPS = 44

def get_train_augmentations():
    augment_pipeline = torch.nn.Sequential(
        T.FrequencyMasking(
            freq_mask_param=int(N_BINS * FREQ_MASK_PARAM)
        ),

        T.TimeMasking(
            time_mask_param=int(N_TIME_STEPS * TIME_MASK_PARAM)
        )
    )

    return augment_pipeline