import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import correlate
from tqdm import tqdm

ORIGINAL_DIR = "../Dataset_orig/Solos"
RECORDED_DIR = "../Dataset/recorded_solos"
ALIGNED_DIR = "../Dataset/solo_data"
HOP_LENGTH = 32

def find_offset_spectral(y_orig, y_rec, sr):
    onset_env_orig = librosa.onset.onset_strength(y=y_orig, sr=sr, hop_length=HOP_LENGTH)
    onset_env_rec = librosa.onset.onset_strength(y=y_rec, sr=sr, hop_length=HOP_LENGTH)

    onset_env_orig = (onset_env_orig - np.mean(onset_env_orig)) / np.std(onset_env_orig)
    onset_env_rec = (onset_env_rec - np.mean(onset_env_rec)) / np.std(onset_env_rec)

    correlation = correlate(onset_env_rec, onset_env_orig, mode='full', method='fft')

    zero_lag_point = len(onset_env_orig) - 1
    lag_frames = np.argmax(correlation) - zero_lag_point

    offset_samples = lag_frames * HOP_LENGTH

    return offset_samples


def main():
    os.makedirs(ALIGNED_DIR, exist_ok=True)

    files_to_process = [f for f in os.listdir(ORIGINAL_DIR) if f.lower().endswith('.wav')]

    print(f"Found {len(files_to_process)} files to align using cross-correlation.")

    for filename in tqdm(files_to_process, desc="Aligning files"):
        original_path = os.path.join(ORIGINAL_DIR, filename)
        recorded_path = os.path.join(RECORDED_DIR, filename)
        aligned_path = os.path.join(ALIGNED_DIR, filename)

        y_orig, sr_orig = librosa.load(original_path, sr=None, mono=True)
        y_rec_unaligned, sr_rec = librosa.load(recorded_path, sr=None, mono=True)

        start_index = find_offset_spectral(y_orig, y_rec_unaligned, sr_orig)
        y_rec_true_amplitude, _ = sf.read(recorded_path, dtype='float32')

        if y_rec_true_amplitude.ndim > 1:
            y_rec_true_amplitude = np.mean(y_rec_true_amplitude, axis=1)

        end_index = start_index + len(y_orig)

        if start_index < 0:
            start_index = 0

        if end_index > len(y_rec_true_amplitude):
            final_aligned_audio = y_rec_true_amplitude[start_index:]
            padding_needed = len(y_orig) - len(final_aligned_audio)
            final_aligned_audio = np.pad(final_aligned_audio, (0, padding_needed), 'constant')
        else:
            final_aligned_audio = y_rec_true_amplitude[start_index:end_index]

        sf.write(aligned_path, final_aligned_audio, sr_orig)

if __name__ == "__main__":
    main()