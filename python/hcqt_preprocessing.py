import pandas as pd
import json
import h5py
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from skmultilearn.model_selection import iterative_train_test_split
from hcqt import load_audio_file, segment_and_compute_hcqt
from tqdm import tqdm
from collections import OrderedDict
from data_processing import *
import multiprocessing as mp
import numpy as np
from augmentations import get_train_augmentations
from print_functions import *
from config import *

class InstrumentDataset:
    INSTRUMENTS = INSTRUMENTS_LABELS
    INSTRUMENT_TO_IDX = {inst: idx for idx, inst in enumerate(INSTRUMENTS)}

    def __init__(self, audio_dir, label_dir, audio_files=None, cache_dir=CACHE_DIR,
                 frame_size_ms=FRAME_SIZE_MS, hop_size_ms=HOP_SIZE_MS, n_harmonics=N_HARMONICS,
                 sample_rate=SAMPLE_RATE, use_cache=USE_CACHE, label_smoothing=LABEL_SMOOTHING,
                 n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, hop_length=CQT_HOP, n_processes=None, augment_fn = None):

        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.cache_dir = cache_dir
        self.frame_size_ms = frame_size_ms
        self.hop_size_ms = hop_size_ms
        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.use_cache = use_cache
        self.label_smoothing = label_smoothing
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.hop_length = hop_length
        self.cqt_frame_duration_s = hop_length / sample_rate
        self.cqt_frames_per_audio_frame = int(np.ceil(frame_size_ms / 1000.0 / self.cqt_frame_duration_s))
        self.n_processes = n_processes if n_processes is not None else max(1, mp.cpu_count() - 1)

        self.augment_fn = augment_fn
        if self.augment_fn:
            print(f"Applying data augmentation to this dataset.")

        self.hdf5_path = os.path.join(self.cache_dir, 'instrument_data.h5')
        self.mode = None
        self.h5_file = None

        if self.use_cache and os.path.exists(self.hdf5_path):
            self.mode = 'hdf5'
            print(f"HDF5 cache found. Initializing in HDF5 mode: {self.hdf5_path}")
            self._init_from_hdf5()
        else:
            self.mode = 'files'
            print(f"HDF5 cache not found. Initializing in 'files' mode to generate cache.")
            if self.use_cache:
                os.makedirs(self.cache_dir, exist_ok=True)

            if audio_files is None:
                self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
            else:
                self.audio_files = audio_files

            self.audio_files.sort()
            print(f"Found {len(self.audio_files)} audio files")
            print(f"Using {self.n_processes} processes for computation")
            if label_smoothing > 0:
                print(f"Using label smoothing: {label_smoothing}")

            self.precompute_all_caches()

            self.file_metadata = {}
            self.memmap_cache = OrderedDict()
            self.metadata_cache = OrderedDict()
            self.max_memmap_files = 10
            self.total_frames = 0
            self.frame_index_map = []
            self.build_index()

    def _init_from_hdf5(self):
        with h5py.File(self.hdf5_path, 'r') as f:
            self.total_frames = len(f['hcqt'])
        print(f"Loaded {self.total_frames} frames from HDF5 cache.")

    def _process_single_file(self, wav_file):
        hcqt_path = self.get_hcqt_cache_path(wav_file)
        metadata_path = self.get_metadata_cache_path(wav_file)

        if self.use_cache and os.path.exists(hcqt_path) and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    json.load(f)
                return wav_file, True, "Cache exists"
            except:
                pass

        try:
            audio_path = os.path.join(self.audio_dir, wav_file)
            y, sr = load_audio_file(audio_path, sample_rate=self.sample_rate)
            if y is None or len(y) == 0:
                return wav_file, False, "Audio file is empty or could not be loaded"

            frame_hcqts, frame_start_times = segment_and_compute_hcqt(
                y=y, sample_rate=sr, frame_size_ms=self.frame_size_ms,
                hop_size_ms=self.hop_size_ms, n_harmonics=self.n_harmonics,
                cqt_hop=self.hop_length, n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave
            )

            if not frame_hcqts:
                return wav_file, False, "No frames extracted from audio."

            labels_df = self.load_labels(wav_file)
            frame_activity_labels = self.create_activity_labels(labels_df, frame_start_times)

            if self.use_cache:
                try:
                    hcqt_array = np.array(frame_hcqts)
                    np.save(hcqt_path, hcqt_array)

                    metadata = {
                        'frame_activity_labels': frame_activity_labels.tolist(),
                        'frame_start_times': np.array(frame_start_times).tolist(),
                        'shape': hcqt_array.shape
                    }
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f)
                    return wav_file, True, "Processed and cached"
                except Exception as e:
                    return wav_file, False, f"Cache save error: {e}"
            return wav_file, True, "Processed (no cache)"
        except Exception as e:
            return wav_file, False, f"Processing error: {e}"

    def precompute_all_caches(self):
        self._precompute_parallel() if self.n_processes > 1 else self._precompute_sequential()

    def _precompute_sequential(self):
        for wav_file in tqdm(self.audio_files, desc="Creating cache files - sequential processing"):
            result = self._process_single_file(wav_file)
            if not result[1]:
                print(f"Failed to process {result[0]}: {result[2]}")

    def _precompute_parallel(self):
        print(f"Parallel cache computation with {self.n_processes} processes.")
        with mp.Pool(processes=self.n_processes) as pool:
            results = list(tqdm(
                pool.imap(self._process_single_file, self.audio_files),
                total=len(self.audio_files),
                desc="Creating cache files using parallel processing"
            ))
        successful = sum(1 for _, success, _ in results if success)
        failed = len(results) - successful
        print(f"Cache computation completed: {successful} successful, {failed} failed")
        for wav_file, success, message in results:
            if not success:
                print(f"Failed {wav_file}: {message}")

    def build_index(self):
        for wav_file in tqdm(self.audio_files, desc="Indexing cached files: "):
            metadata = self.get_metadata(wav_file)
            if metadata and 'shape' in metadata and metadata['shape']:
                frame_count = metadata['shape'][0]
            else:
                continue
            self.file_metadata[wav_file] = {'frame_count': frame_count}
            for frame_idx in range(frame_count):
                self.frame_index_map.append((wav_file, frame_idx))
            self.total_frames += frame_count
        print(f"Total indexed frames: {self.total_frames}")

    def map_index_to_file(self, idx):
        if idx >= len(self.frame_index_map):
            raise IndexError(f"Index {idx} out of range")

        return self.frame_index_map[idx]

    def get_hcqt_cache_path(self, filename):
        base_name = filename.replace('.wav', '')
        param_str = f"{base_name}_hcqt_" + CACHE_NAME
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

        return os.path.join(self.cache_dir, f"hcqt_{base_name}_{param_hash}.npy")

    def get_metadata_cache_path(self, filename):
        base_name = filename.replace('.wav', '')
        param_str = f"{base_name}_hcqt_" + CACHE_NAME
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

        return os.path.join(self.cache_dir, f"metadata_{base_name}_{param_hash}.json")

    def load_labels(self, wav_filename):
        base_name = wav_filename.replace('.wav', '')
        label_file = f"converted_{base_name}.csv"
        label_path = os.path.join(self.label_dir, label_file)

        try:
            return pd.read_csv(label_path)
        except Exception as e:
            print(f"Error loading {label_path}: {e}")
            return pd.DataFrame()

    def create_activity_labels(self, labels_df, frame_times):
        num_frames = len(frame_times)
        num_time_steps = self.cqt_frames_per_audio_frame
        activity_labels = np.zeros((num_frames, len(self.INSTRUMENTS), num_time_steps), dtype=np.float32)

        if 'instrument_name' not in labels_df.columns:
            return self.apply_label_smoothing(activity_labels)

        start_times = labels_df['start_seconds'].values
        end_times = labels_df['end_seconds'].values
        instruments = labels_df['instrument_name'].values
        valid_mask = np.array([inst in self.INSTRUMENT_TO_IDX for inst in instruments])
        if not np.any(valid_mask):
            return self.apply_label_smoothing(activity_labels)

        valid_start_times = start_times[valid_mask]
        valid_end_times = end_times[valid_mask]
        valid_instruments = instruments[valid_mask]
        valid_inst_indices = np.array([self.INSTRUMENT_TO_IDX[inst] for inst in valid_instruments])

        for frame_idx, frame_start_time in enumerate(frame_times):
            for cqt_step in range(num_time_steps):
                cqt_time = frame_start_time + (cqt_step * self.cqt_frame_duration_s)
                active_mask = (valid_start_times <= cqt_time) & (valid_end_times > cqt_time)
                if np.any(active_mask):
                    active_inst_indices = valid_inst_indices[active_mask]
                    activity_labels[frame_idx, active_inst_indices, cqt_step] = 1.0
        return self.apply_label_smoothing(activity_labels)

    def apply_label_smoothing(self, labels):
        if self.label_smoothing == 0.0:
            return labels
        n_classes = len(self.INSTRUMENTS)
        positive_value = 1.0 - self.label_smoothing + (self.label_smoothing / n_classes)
        negative_value = self.label_smoothing / n_classes
        return np.where(labels == 1.0, positive_value, negative_value).astype(np.float32)

    def get_memmap_array(self, wav_filename):
        if wav_filename in self.memmap_cache:
            self.memmap_cache.move_to_end(wav_filename)
            return self.memmap_cache[wav_filename]
        hcqt_path = self.get_hcqt_cache_path(wav_filename)
        if (self.get_metadata(wav_filename) is None) or (not os.path.exists(hcqt_path)):
            return None
        try:
            mmap_array = np.load(hcqt_path, mmap_mode='r')
            self.memmap_cache[wav_filename] = mmap_array
            if len(self.memmap_cache) > self.max_memmap_files:
                del self.memmap_cache[next(iter(self.memmap_cache))]
            return mmap_array
        except Exception as e:
            print(f"Error loading memory-mapped array for {wav_filename}: {e}")
            return None

    def get_metadata(self, wav_filename):
        if wav_filename in self.metadata_cache:
            self.metadata_cache.move_to_end(wav_filename)
            return self.metadata_cache[wav_filename]
        metadata_path = self.get_metadata_cache_path(wav_filename)
        try:
            if not os.path.exists(metadata_path):
                return None
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.metadata_cache[wav_filename] = metadata
            if len(self.metadata_cache) > self.max_memmap_files:
                del self.metadata_cache[next(iter(self.metadata_cache))]
            return metadata
        except Exception as e:
            print(f"Error loading metadata for {wav_filename}: {e}")
            return None

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        if self.mode == 'hdf5':
            try:
                if self.h5_file is None:
                    self.h5_file = h5py.File(self.hdf5_path, 'r')

                hcqt_data = self.h5_file['hcqt'][idx]
                activity_label_data = self.h5_file['activity_label'][idx]
                wav_filename = self.h5_file['wav_filename'][idx].decode('utf-8')
                frame_start_time = self.h5_file['frame_start_time'][idx]

                hcqt = torch.from_numpy(hcqt_data)
                activity_label = torch.from_numpy(activity_label_data)

                if self.augment_fn:
                    hcqt = self.augment_fn(hcqt)

                return hcqt, activity_label, wav_filename, frame_start_time
            except Exception as e:
                print(f"\n--- HDF5 DataLoader Worker Error ---")
                print(f"FATAL: Failed to process item at index: {idx} from {self.hdf5_path}")
                print(f"  -> Error Type: {type(e).__name__}")
                print(f"  -> Error Details: {e}")
                raise e
        else:
            wav_filename, frame_idx = None, None
            try:
                wav_filename, frame_idx = self.map_index_to_file(idx)
                mmap_array = self.get_memmap_array(wav_filename)
                if mmap_array is None:
                    raise IOError(f"Could not load mmap array for {wav_filename}")

                metadata = self.get_metadata(wav_filename)
                if metadata is None:
                    raise IOError(f"Could not load metadata for {wav_filename}")

                hcqt_data = mmap_array[frame_idx].copy()
                hcqt_data_transposed = np.transpose(hcqt_data, (0, 2, 1))
                hcqt = torch.FloatTensor(hcqt_data_transposed)

                if self.augment_fn:
                    hcqt = self.augment_fn(hcqt)

                single_frame_label = metadata['frame_activity_labels'][frame_idx]
                activity_label = torch.FloatTensor(single_frame_label)
                frame_start_time = metadata['frame_start_times'][frame_idx]

                return hcqt, activity_label, wav_filename, frame_start_time
            except Exception as e:
                print(f"\nFiles DataLoader Worker Error")
                print(f"FATAL: Failed to process item at global index: {idx}")
                if wav_filename and frame_idx is not None:
                    print(f"File: {wav_filename}, Frame Index: {frame_idx}")
                print(f"Error Type: {type(e).__name__}")
                print(f"Error Details: {e}")
                raise e

    @property
    def activity_labels(self):
        if self.mode == 'hdf5':
            with h5py.File(self.hdf5_path, 'r') as f:
                return f['activity_label'][:]
        else:
            all_labels = []
            for wav_file in self.audio_files:
                metadata = self.get_metadata(wav_file)
                if metadata and 'frame_activity_labels' in metadata:
                    all_labels.append(np.array(metadata['frame_activity_labels']))
            return np.vstack(all_labels) if all_labels else np.array([])

def get_activity_label_statistics(dataset, dataset_name):
    print(f"\n{dataset_name} Label Statistics:")
    print("-" * 50)
    activity_labels = dataset.activity_labels

    if activity_labels.size == 0:
        print("Dataset is empty. No statistics to show.")
        return
    print(f"Total frames: {len(activity_labels)}")
    print(f"Time steps per frame: {activity_labels.shape[2]}")

    for i, instrument in enumerate(INSTRUMENTS_LABELS):
        total_active_steps = np.sum(activity_labels[:, i, :])
        total_possible_steps = activity_labels.shape[0] * activity_labels.shape[2]
        prevalence = total_active_steps / total_possible_steps
        frames_with_activity = np.sum(np.any(activity_labels[:, i, :], axis=1))
        frame_prevalence = frames_with_activity / len(activity_labels)
        print(f"{instrument}: {total_active_steps:.0f} active steps ({prevalence:.6f} step prevalence)")
        print(f"  {frames_with_activity:.0f} frames with activity ({frame_prevalence:.4f} frame prevalence)")

def convert_to_hdf5(dataset, hdf5_path):
    print(f"Converting dataset to HDF5 format at {hdf5_path}.")
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

    if not len(dataset) > 0:
        print("Warning: Dataset is empty. Skipping HDF5 creation.")
        return

    first_hcqt, first_label, _, _ = dataset[0]
    hcqt_shape = (len(dataset),) + first_hcqt.shape
    label_shape = (len(dataset),) + first_label.shape

    with h5py.File(hdf5_path, 'w') as f:
        dset_hcqt = f.create_dataset('hcqt', shape=hcqt_shape, dtype=np.float32,
                                     chunks=(1,) + first_hcqt.shape, compression='gzip')
        dset_label = f.create_dataset('activity_label', shape=label_shape, dtype=np.float32,
                                      chunks=(1,) + first_label.shape, compression='gzip')
        dset_time = f.create_dataset('frame_start_time', shape=(len(dataset),), dtype=np.float64)

        dt = h5py.special_dtype(vlen=str)
        dset_fname = f.create_dataset('wav_filename', shape=(len(dataset),), dtype=dt)

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        current_idx = 0
        for hcqt_batch, label_batch, fname_batch, time_batch in tqdm(loader, desc="Writing to HDF5"):
            batch_len = len(hcqt_batch)
            dset_hcqt[current_idx: current_idx + batch_len] = hcqt_batch.numpy()
            dset_label[current_idx: current_idx + batch_len] = label_batch.numpy()
            dset_time[current_idx: current_idx + batch_len] = time_batch.numpy()
            dset_fname[current_idx: current_idx + batch_len] = fname_batch
            current_idx += batch_len

    print(f"HDF5 conversion complete. {len(dataset)} items saved.")

def prepare_multilabel_matrix(file_list, label_dir, instrument_labels):
    print("Preparing multi-label matrix for stratification.")

    instrument_to_idx = {inst: i for i, inst in enumerate(instrument_labels)}
    num_files = len(file_list)
    num_instruments = len(instrument_labels)
    y = np.zeros((num_files, num_instruments), dtype=int)

    for i, wav_filename in enumerate(tqdm(file_list, desc="Scanning labels for stratification")):
        base_name = wav_filename.replace('.wav', '')

        label_file = f"converted_{base_name}.csv"
        label_path = os.path.join(label_dir, label_file)

        try:
            if os.path.exists(label_path):
                labels_df = pd.read_csv(label_path)
                present_instruments = set(labels_df['instrument_name'].unique())
                for inst in present_instruments:
                    if inst in instrument_to_idx:
                        j = instrument_to_idx[inst]
                        y[i, j] = 1

        except Exception as e:
            print(f"Warning: Could not process label file {label_path}: {e}")

    return y

def create_datasets(cache_dir=CACHE_DIR,
                    label_smoothing=LABEL_SMOOTHING,
                    n_processes=1,
                    **dataset_kwargs):
    datasets = {}
    train_augmentations = get_train_augmentations()

    dataset_configs = {
        'train': {
            'audio_dir': TRAIN_AUDIO_DIR,
            'label_dir': TRAIN_LABEL_DIR,
            'label_smoothing': label_smoothing,
            'augment_fn': train_augmentations
        },
        'val': {
            'audio_dir': VAL_AUDIO_DIR,
            'label_dir': VAL_LABEL_DIR,
            'label_smoothing': 0.0,
            'augment_fn': None
        },
        'test': {
            'audio_dir': TEST_AUDIO_DIR,
            'label_dir': TEST_LABEL_DIR,
            'label_smoothing': 0.0,
            'augment_fn': None
        }
    }

    for name, config in dataset_configs.items():
        print(f"\nProcessing '{name}' dataset.")
        current_cache_dir = os.path.join(cache_dir, name)
        hdf5_path = os.path.join(current_cache_dir, 'instrument_data.h5')

        audio_files = [f for f in os.listdir(config['audio_dir']) if f.endswith('.wav')]
        print(f"Found {len(audio_files)} files in {config['audio_dir']}")

        if not os.path.exists(hdf5_path):
            print(f"HDF5 cache not found for '{name}'. Pre-computing and creating it.")
            temp_dataset = InstrumentDataset(
                audio_dir=config['audio_dir'],
                label_dir=config['label_dir'],
                audio_files=audio_files,
                cache_dir=current_cache_dir,
                use_cache=True,
                label_smoothing=config['label_smoothing'],
                n_processes=n_processes,
                **dataset_kwargs
            )
            convert_to_hdf5(temp_dataset, hdf5_path)
            del temp_dataset

        print(f"Loading '{name}' dataset from HDF5 cache.")
        dataset = InstrumentDataset(
            audio_dir=config['audio_dir'],
            label_dir=config['label_dir'],
            audio_files=audio_files,
            cache_dir=current_cache_dir,
            use_cache=True,
            label_smoothing=config['label_smoothing'],
            n_processes=n_processes,
            augment_fn=config['augment_fn'],
            **dataset_kwargs
        )
        datasets[name] = dataset

    return datasets['train'], datasets['val'], datasets['test']

if __name__ == "__main__":
    try:
        train_dataset, val_dataset, test_dataset = create_datasets(
            label_smoothing=LABEL_SMOOTHING,
            n_processes=max(1, NUM_WORKERS)
        )

        get_activity_label_statistics(train_dataset, "Training")
        get_activity_label_statistics(val_dataset, "Validation")
        get_activity_label_statistics(test_dataset, "Testing")

        print(f"\n{'=' * 60}")
        print(f"DATASET SUMMARY")
        print(f"{'=' * 60}")
        print(f"Training samples:   {len(train_dataset):,}")
        print(f"Validation samples: {len(val_dataset):,}")
        print(f"Testing samples:    {len(test_dataset):,}")
        print(f"Total samples:      {len(train_dataset) + len(val_dataset) + len(test_dataset):,}")
        print(f"Dataset creation completed successfully!")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()