"""
Dataset statistics (with a pickle cache), class weights for the loss and the per-instrument
F1-optimal decision thresholds.
"""
import torch
import numpy as np
import time
from datetime import datetime
import hashlib
from instrument_detection.config import *
import pickle

def generate_dataset_hash(train_dataset, validation_dataset, test_dataset):
    """
    Builds a short id from the three dataset sizes, the label smoothing, the instrument list and the frames per audio frame.
    The id names the statistics cache files, so a changed dataset gets a new cache.
    """
    hash_elements = [
        len(train_dataset),
        len(validation_dataset),
        len(test_dataset),
        str(getattr(train_dataset, 'label_smoothing', 0)),
        str(getattr(train_dataset, 'INSTRUMENTS', [])),
        str(getattr(train_dataset, 'cqt_frames_per_audio_frame', 44))
    ]
    hash_string = ''.join(map(str, hash_elements))

    return hashlib.md5(hash_string.encode()).hexdigest()[:16]

def load_dataset_statistics(metadata_file):
    """
    Loads the cached statistics from the pickle file if it exists.
    Ignores the cache if the instruments, number of instruments or label smoothing differ from the current config.
    Returns (None, None) when there is no usable cache.
    """
    try:
        if not os.path.exists(metadata_file):
            return None, None

        with open(metadata_file, 'rb') as f:
            cache_data = pickle.load(f)

        config_snapshot = cache_data.get('config_snapshot', {})

        if (config_snapshot.get('instruments') != INSTRUMENTS_LABELS or
                config_snapshot.get('num_instruments') != NUM_INSTRUMENTS or
                config_snapshot.get('label_smoothing') != LABEL_SMOOTHING):
            return None, None

        statistics = cache_data['statistics']
        target_statistics = {
            'temporal_mean': np.array(cache_data['target_statistics']['temporal_mean']),
            'temporal_sum': np.array(cache_data['target_statistics']['temporal_sum']),
            'total_samples': cache_data['target_statistics']['total_samples']
        }
        cache_age = datetime.now() - datetime.fromisoformat(cache_data['timestamp'])
        print(f"Loaded cached dataset statistics (age: {cache_age.days} days)")

        return statistics, target_statistics

    except Exception as e:
        print(f"Warning: Could not load dataset statistics cache: {e}")

        return None, None

def save_dataset_statistics(temporal_stats, target_stats, metadata_file):
    """
    Saves the statistics to a pickle file together with a timestamp and a snapshot of the config.
    Converts numpy arrays to lists first.
    Returns True on success and False if the write fails.
    """
    try:
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)

        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': temporal_stats,
            'target_statistics': {
                'temporal_mean': target_stats['temporal_mean'].tolist() if hasattr(target_stats['temporal_mean'],'tolist') else target_stats['temporal_mean'],
                'temporal_sum': target_stats['temporal_sum'].tolist() if hasattr(target_stats['temporal_sum'],'tolist') else target_stats['temporal_sum'],
                'total_samples': target_stats['total_samples']
            },
            'config_snapshot': {
                'instruments': INSTRUMENTS_LABELS,
                'num_instruments': NUM_INSTRUMENTS,
                'label_smoothing': LABEL_SMOOTHING,
            }
        }

        with open(metadata_file, 'wb') as f:
            pickle.dump(cache_data, f)

        print(f"Dataset statistics cached to: {metadata_file}")
        return True

    except Exception as e:
        print(f"Warning: Could not save dataset statistics cache: {e}")

        return False

def compute_dataset_statistics(data_loader, dataset_hash, device,  phase_name="Training"):
    """
    Picks the cache file for the phase (Training, Validation, Testing or other).
    Returns the cached statistics if they exist.
    Otherwise computes them with analyze_data_statistics and saves them to the cache.
    """
    cache_dir = './dataset_metadata'
    if phase_name == "Training":
        metadata_file = os.path.join(cache_dir, f'dataset_stats_{dataset_hash}.pkl')
    elif phase_name == "Validation":
        metadata_file = os.path.join(cache_dir, f'dataset_stats_{dataset_hash}_val.pkl')
    elif phase_name == "Testing":
        metadata_file = os.path.join(cache_dir, f'dataset_stats_{dataset_hash}_test.pkl')
    else:
        metadata_file = os.path.join(cache_dir, f'dataset_stats_{dataset_hash}_{phase_name.lower()}.pkl')

    print(f"Dataset hash: {dataset_hash} ({phase_name})")
    statistics, target_statistics = load_dataset_statistics(metadata_file)

    if statistics is not None and target_statistics is not None:
        print(f"Using cached dataset statistics for {phase_name}")

        return statistics, target_statistics

    print(f"Computing {phase_name} dataset statistics...")
    statistics, target_statistics = analyze_data_statistics(data_loader, phase_name, device)
    save_dataset_statistics(statistics, target_statistics, metadata_file)

    return statistics, target_statistics

def calculate_class_weights(statistics, device):
    """
    Clips each instrument's prevalence to between 0.001 and 0.999.
    Computes positive class weights as (1 - prevalence) / prevalence for the loss, so rare instruments count more.
    Computes a starting decision threshold per instrument, higher for rarer instruments.
    """
    print(f"statistics['mean']: {statistics['mean']} (shape: {np.array(statistics['mean']).shape})")

    prevalence = statistics['mean']
    prevalence = np.clip(prevalence, 0.001, 0.999)

    pos_weights = torch.FloatTensor((1 - prevalence) / prevalence).to(device)
    thresholds = np.clip(0.5 - 0.3 * (prevalence - 0.5), 0.1, 0.9)

    print(f"  Prevalence: {prevalence}")
    print(f"  Pos_weights: {pos_weights}")
    print(f"  Thresholds: {thresholds}")

    return {
        'pos_weights': pos_weights,
        'thresholds': thresholds
    }


def find_optimal_thresholds(y_true, y_pred):
    """
    Finds the optimal F1-score threshold for each instrument class to three decimal places.
    """
    best_thresholds = np.zeros(y_true.shape[1])

    for i in range(y_true.shape[1]):
        best_f1 = -1.0
        best_thresh = 0.5

        true_flat = y_true[:, i]
        pred_flat = y_pred[:, i]

        for threshold in np.arange(0.001, 1.0, 0.001):
            pred_binary = (pred_flat > threshold).astype(int)

            tp = np.sum((pred_binary == 1) & (true_flat == 1))
            fp = np.sum((pred_binary == 1) & (true_flat == 0))
            fn = np.sum((pred_binary == 0) & (true_flat == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            if f1 >= best_f1:
                best_f1 = f1
                best_thresh = threshold

        best_thresholds[i] = best_thresh
        instrument_name = INSTRUMENTS_LABELS[i] if i < len(INSTRUMENTS_LABELS) else f"Unknown ({i})"
        print(f"  Instrument {i} ({instrument_name}): Optimal Threshold = {best_thresh:.3f} (F1 = {best_f1:.4f})")

    print("\nOptimal thresholds found:")
    print(np.round(best_thresholds, 3))

    return best_thresholds