import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from config import *

def load_audio_file(file_path, sample_rate=SAMPLE_RATE):
    y, sample_rate = librosa.load(file_path, sr=sample_rate, mono=True)
    return y, sample_rate


def compute_cqt(y, sample_rate, hop_length, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, fmin=None):
    if fmin is None:
        fmin = librosa.note_to_hz('C1')

    return librosa.cqt(
        y=y,
        sr=sample_rate,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin
    )

def compute_hcqt(y, sample_rate, n_harmonics=N_HARMONICS, hop_length=HOP_LENGTH,
                 n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, fmin=None):
    if fmin is None:
        fmin = librosa.note_to_hz('C1')

    hcqt_list = []

    for h in range(1, n_harmonics + 1):
        harmonic_fmin = fmin * h
        cqt = compute_cqt(
            y=y,
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            fmin=harmonic_fmin
        )

        hcqt_list.append(np.abs(cqt))

    hcqt = np.stack(hcqt_list, axis=0)

    frame_times = librosa.frames_to_time(
        frames=np.arange(hcqt.shape[-1]),
        sr=sample_rate,
        hop_length=hop_length
    )

    return hcqt, frame_times


def extract_frames_from_hcqt(hcqt, frame_times, frame_size_ms, hop_size_ms, sample_rate):
    frame_size_sec = frame_size_ms / 1000.0
    hop_size_sec = hop_size_ms / 1000.0

    cqt_frame_rate = len(frame_times) / (frame_times[-1])
    frames_per_window = int(np.ceil(frame_size_sec * cqt_frame_rate))
    frames_per_hop = int(np.ceil(hop_size_sec * cqt_frame_rate))

    n_harmonics, n_bins, total_frames = hcqt.shape
    num_windows = int((total_frames - frames_per_window) // frames_per_hop + 1)

    frame_hcqts = []
    frame_start_times = []

    for i in range(num_windows):
        start_frame = i * frames_per_hop
        end_frame = start_frame + frames_per_window

        window_hcqt = hcqt[:, :, start_frame:end_frame]
        frame_hcqts.append(window_hcqt)

        if start_frame >= len(frame_times):
            start_time = start_frame / cqt_frame_rate
        else:
            start_time = frame_times[start_frame]

        frame_start_times.append(start_time)

    return frame_hcqts, frame_start_times

def segment_and_compute_hcqt(y, sample_rate, frame_size_ms, hop_size_ms, n_harmonics=N_HARMONICS,
                             cqt_hop=CQT_HOP, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, fmin=None):
    full_hcqt, hcqt_frame_times = compute_hcqt(
        y, sample_rate, n_harmonics=n_harmonics, hop_length=cqt_hop,
        n_bins=n_bins, bins_per_octave=bins_per_octave, fmin=fmin
    )

    frame_hcqts, frame_start_times = extract_frames_from_hcqt(
        full_hcqt, hcqt_frame_times, frame_size_ms, hop_size_ms, sample_rate
    )

    return frame_hcqts, frame_start_times

def visualize_frame(frame_hcqt, frame_idx, start_time, save_dir=None):
    plt.figure(figsize=(12, 8))

    combined_hcqt = np.sum(frame_hcqt, axis=0)

    img = librosa.display.specshow(
        librosa.amplitude_to_db(combined_hcqt, ref=np.max),
        sr=44100,
        x_axis='time',
        y_axis='cqt_note',
        bins_per_octave=BINS_PER_OCTAVE
    )
    plt.title(f"Frame number {frame_idx + 1} : start_time {start_time:.2f}s")
    plt.colorbar(img, format='%+2.0f dB')
    plt.tight_layout()

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig(os.path.join(save_dir, f"frame_{frame_idx + 1}.png"))
        plt.close()
    else:
        plt.show()


def visualize_frames(frame_hcqts, frame_times, save_dir=None, max_frames=None):
    if max_frames is None:
        frames_to_vis = len(frame_hcqts)
    else:
        frames_to_vis = min(max_frames, len(frame_hcqts))

    for frame_idx in range(frames_to_vis):
        visualize_frame(
            frame_hcqts[frame_idx],
            frame_idx,
            frame_times[frame_idx],
            save_dir
        )


def process_song(file_path, frame_size_ms, hop_size_ms, n_harmonics=N_HARMONICS):
    y, sample_rate = load_audio_file(file_path)

    pad_duration_ms = frame_size_ms
    pad_samples = int((pad_duration_ms / 1000.0) * sample_rate)
    y_padded = np.pad(y, (pad_samples, pad_samples), mode='reflect')

    frame_hcqts, frame_times = segment_and_compute_hcqt(
        y=y_padded,
        sample_rate=sample_rate,
        frame_size_ms=frame_size_ms,
        hop_size_ms=hop_size_ms,
        n_harmonics=n_harmonics
    )

    return frame_hcqts, sample_rate, frame_times


def ms_to_samples(ms, sample_rate):
    return int((ms / 1000.0) * sample_rate)


def main():
    musicnet_file = "./Dataset/Testing/1791.wav"
    save_dir = "hcqt_original"

    frame_size_ms = 500
    hop_size_ms = 500
    n_harmonics = 5
    max_frames_to_display = 20

    print(f"Processing file: {musicnet_file}")
    print(f"Parameters: frame_size={frame_size_ms} ms, hop_size={hop_size_ms} ms, n_harmonics={n_harmonics}")

    frame_hcqts, sr, frame_times = process_song(
        file_path=musicnet_file,
        frame_size_ms=frame_size_ms,
        hop_size_ms=hop_size_ms,
        n_harmonics=n_harmonics
    )

    frame_size_samples = ms_to_samples(frame_size_ms, sr)
    hop_size_samples = ms_to_samples(hop_size_ms, sr)
    print(f"Frame size: {frame_size_ms} ms ({frame_size_samples} samples)")
    print(f"Hop size: {hop_size_ms} ms ({hop_size_samples} samples)")
    print(f"Processed {len(frame_hcqts)} frames")

    if len(frame_hcqts) > 0:
        visualize_frames(frame_hcqts, frame_times, save_dir, max_frames=max_frames_to_display)
        print(f"Frame visualizations saved to: {save_dir}/")


if __name__ == "__main__":
    main()