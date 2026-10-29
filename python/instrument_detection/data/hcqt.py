"""
Loads a recording, computes its HCQT and cuts it into fixed windows.
Also has a small demo in main() that saves the first windows of one test file as images.
"""
import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
from instrument_detection.config import *

def load_audio_file(file_path, sample_rate=SAMPLE_RATE):
    """
    Parameters:
    -----------
    file_path : str
        Path to the audio file
    sample_rate : int
        Target sampling rate

    Returns:
    --------
    y : np.ndarray
        Audio time series
    sample_rate : int
        Sampling rate
    """
    try:
        y, sample_rate = librosa.load(file_path, sr=sample_rate, mono=True)
        return y, sample_rate

    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None


def compute_cqt(y, sample_rate, hop_length, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, fmin=None):
    """
    Parameters:
    -----------
    y : np.ndarray
        Audio time series
    sample_rate : int
        Sampling rate
    hop_length : int
        Number of samples between successive frames
    n_bins : int
        Number of frequency bins
    bins_per_octave : int
        Number of bins per octave
    fmin : float or None
        Minimum frequency. If None, defaults to C1 (~32.7 Hz)

    Returns:
    --------
    CQT : np.ndarray
        Constant-Q transform
    """
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
    """
    Parameters:
    -----------
    y : np.ndarray
        Audio time series
    sample_rate : int
        Sampling rate
    n_harmonics : int
        Number of harmonics to compute (including fundamental)
    hop_length : int
        Number of samples between successive frames
    n_bins : int
        Number of frequency bins
    bins_per_octave : int
        Number of bins per octave
    fmin : float or None
        Minimum frequency for the fundamental. If None, defaults to C1 (~32.7 Hz)

    Returns:
    --------
    HCQT : np.ndarray
        Harmonic Constant-Q transform with shape (n_harmonics, n_bins, n_frames)
    frame_times : np.ndarray
        Time stamps for each frame in seconds
    """
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
    """
    Extract overlapping frames from full HCQT efficiently.

    Parameters:
    -----------
    hcqt : np.ndarray
        Full HCQT with shape (n_harmonics, n_bins, n_frames)
    frame_times : np.ndarray
        Time stamps for each HCQT frame
    frame_size_ms : float
        Frame size in milliseconds
    hop_size_ms : float
        Hop size in milliseconds
    sample_rate : int
        Audio sample rate

    Returns:
    --------
    frame_hcqts : list
        List of HCQT frames
    frame_start_times : list
        List of start times for each frame in seconds
    """
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
    """
    Compute full HCQT first, then extract frames.

    Parameters:
    -----------
    y : np.ndarray
        Audio time series
    sample_rate : int
        Sampling rate
    frame_size_ms : float
        Frame size in milliseconds
    hop_size_ms : float
        Hop size in milliseconds
    n_harmonics : int
        Number of harmonics for HCQT
    cqt_hop : int
        Hop length for CQT computation
    n_bins : int
        Number of frequency bins
    bins_per_octave : int
        Number of bins per octave
    fmin : float or None
        Minimum frequency for the fundamental

    Returns:
    --------
    frame_hcqts : list
        List of HCQT for each frame
    frame_times : list
        List of start times for each frame in seconds
    """
    full_hcqt, hcqt_frame_times = compute_hcqt(
        y, sample_rate, n_harmonics=n_harmonics, hop_length=cqt_hop,
        n_bins=n_bins, bins_per_octave=bins_per_octave, fmin=fmin
    )

    frame_hcqts, frame_start_times = extract_frames_from_hcqt(
        full_hcqt, hcqt_frame_times, frame_size_ms, hop_size_ms, sample_rate
    )

    return frame_hcqts, frame_start_times

def visualize_frame(frame_hcqt, frame_idx, start_time, save_dir=None):
    """
    Parameters:
    -----------
    frame_hcqt : np.ndarray
        HCQT for a single frame with shape (n_harmonics, n_bins, n_frames)
    frame_idx : int
        Frame index number
    start_time : float
        Start time of the frame in seconds
    save_dir : str or None
        Directory to save the image. If None, just display.
    """
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
    """
    Parameters:
    -----------
    frame_hcqts : list
        List of HCQT for each frame
    frame_times : list
        List of start times for each frame in seconds
    save_dir : str or None
        Directory to save the images. If None, just display.
    max_frames : int or None
        Maximum number of frames to visualize. If None, visualize all frames.
    """
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


def process_musicnet_file(file_path, frame_size_ms, hop_size_ms, n_harmonics=N_HARMONICS):
    """
    Parameters:
    -----------
    file_path : str
        Path to the MusicNet audio file
    frame_size_ms : float
        Frame size in milliseconds
    hop_size_ms : float
        Hop size in milliseconds
    n_harmonics : int
        Number of harmonics for HCQT

    Returns:
    --------
    frame_hcqts : list
        List of HCQT for each frame
    sr : int
        Sampling rate
    frame_times : list
        List of start times for each frame in seconds
    """
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
    """
    Convert milliseconds to samples.
    """
    return int((ms / 1000.0) * sample_rate)


def main():
    """
    Runs the HCQT pipeline on one test recording.
    Splits it into 500 ms windows with no overlap, prints the sizes, then saves images of the first 20 windows.
    """
    musicnet_file = "./Dataset/Testing/1791.wav"
    save_dir = "hcqt_original"

    frame_size_ms = 500
    hop_size_ms = 500
    n_harmonics = 5
    max_frames_to_display = 20

    print(f"Processing file: {musicnet_file}")
    print(f"Parameters: frame_size={frame_size_ms} ms, hop_size={hop_size_ms} ms, n_harmonics={n_harmonics}")

    frame_hcqts, sr, frame_times = process_musicnet_file(
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
