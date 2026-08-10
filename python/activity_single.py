import torch
import numpy as np
import json
from model import build_model
from hcqt import load_audio_file, compute_hcqt, extract_frames_from_hcqt
from config import *
import traceback

CONFIG = {
    'audio_path': './Dataset/Testing/test_data/2082.wav',
    'model_path': './models/instrument_model_2025-10-16_13-42-05.pth',
    'output_file': None,
    'verbose': True,
    'device': None,
    'batch_size': 32,
    'inference_hop_ms': 500
}

class ActivityDetector:
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.thresholds = self.get_thresholds()
        self.model.eval()

    def load_model(self):
        """
        Function to load a model. It checks if the file exists at a given path, and if so, loads the model
        using pytorch load on the target device.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model = build_model().to(self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        
        return model

    def get_thresholds(self):
        """
        Loads the saved checkpoint from the model. Looks for possible dictonary entries under which the
        instrument thresholds may be saved under. If found, uses those thresholds. If no thresholds are found, uses
        0.5 as a default.
        """
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        if 'weights_info' in checkpoint and 'temporal_thresholds' in checkpoint['weights_info']:
            thresholds_data = checkpoint['weights_info']['temporal_thresholds']
        elif 'thresholds' in checkpoint:
            thresholds_data = checkpoint['thresholds']
        else:
            thresholds_data = None

        if thresholds_data is not None:
            thresholds = np.array(thresholds_data)
            print(f"Using saved thresholds: {thresholds}")
            return thresholds
        else:
            thresholds = np.full(NUM_INSTRUMENTS, 0.5)
            print(f"Warning: No saved thresholds found, using default: {thresholds}")
            return thresholds

    def create_prediction_windows(self, frame_hcqts):
        """
        Extracts the number of harmonics, frequency bins and time frames from parsed HCQT.
        Transposes the frames so that it is harmonics, time, frequency.
        Pads frames that are not of expected size, trims frames that are not of expected size.
        """
        n_harmonics, freq_bins = frame_hcqts[0].shape[:2]
        time_frames = frame_hcqts[0].shape[2]
        n_windows = len(frame_hcqts)
        windows = np.zeros((n_windows, n_harmonics, time_frames, freq_bins))

        for i, frame_hcqt in enumerate(frame_hcqts):
            window = np.transpose(frame_hcqt, (0, 2, 1))
            patch_frames = window.shape[1]

            if patch_frames < time_frames:
                padding_size = time_frames - patch_frames
                padding = np.zeros((n_harmonics, padding_size, freq_bins))
                window = np.concatenate([window, padding], axis=1)
            elif patch_frames > time_frames:
                window = window[:, :time_frames, :]

            windows[i] = window

        return windows

    def batch_inference(self, windows, batch_size=None):
        """
        Calculates the number of batches using integer division.
        Iterates over each batch, calcuting the indices of the appropriate frames which match the batch.
        Converts the batch to a float tensor on the GPU. Parses the input to the model.
        Applies sigmoid function to model output and transfers back to CPU.
        """
        if batch_size is None:
            batch_size = CONFIG['batch_size']

        all_predictions = []
        num_batches = (len(windows) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(windows))

                batch_data = windows[start_idx:end_idx]
                batch_tensor = torch.FloatTensor(batch_data).to(self.device)

                output = self.model(batch_tensor)
                predictions = torch.sigmoid(output).cpu().numpy()
                all_predictions.append(predictions)

        return np.vstack(all_predictions)

    def aggregate_overlapping_predictions(self, overlapping_preds, total_cqt_frames, hop_ms, pad_samples):
        """
        Extracts the number of patches, instruments and time steps.
        Calculates the number of frames per hop.
        Extracts the corresponding overlapping predicitons for each frame.
        """
        num_windows, num_instruments, window_len_frames = overlapping_preds.shape
        aggregated_preds = [[] for _ in range(total_cqt_frames)]

        cqt_frame_time = (CQT_HOP / SAMPLE_RATE) * 1000
        frames_per_hop = max(1, round(hop_ms / cqt_frame_time))

        for i in range(num_windows):
            start_frame = i * frames_per_hop
            for j in range(window_len_frames):
                current_frame_idx = start_frame + j

                if current_frame_idx < total_cqt_frames:
                    prediction_for_frame = overlapping_preds[i, :, j]
                    aggregated_preds[current_frame_idx].append(prediction_for_frame)

        pad_frames = pad_samples // CQT_HOP
        start_frame_orig = pad_frames
        end_frame_orig = total_cqt_frames - pad_frames

        return aggregated_preds[start_frame_orig:end_frame_orig]

    def process_audio(self, audio_path, output_file=None, verbose=True):
        """
        Loads the audio file at the parsed path.
        Applies reflection padding to the beginning of the audio file.
        Computes the HCQT of the audio file and then extracts the frames from the HCQT.
        Performs inference on the frames and aggregates/averages overlapping predictions.
        """
        print("=" * 80)
        y, sr = load_audio_file(audio_path, sample_rate=SAMPLE_RATE)

        if y is None:
            raise ValueError(f"Failed to load audio file: {audio_path}")

        pad_duration_ms = FRAME_SIZE_MS / 2.0
        pad_samples = int((pad_duration_ms / 1000.0) * sr)
        y_padded = np.pad(y, (pad_samples, pad_samples), mode='reflect')

        inference_hop_ms = CONFIG['inference_hop_ms']
        print(f"Using hop size: {inference_hop_ms:.2f} ms")

        hcqt, frame_times = compute_hcqt(
            y=y_padded,
            sample_rate=sr,
            n_harmonics=N_HARMONICS,
            hop_length=CQT_HOP,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE
        )
        num_cqt_frames = hcqt.shape[-1]

        frame_hcqts, frame_start_times = extract_frames_from_hcqt(
            hcqt=hcqt,
            frame_times=frame_times,
            frame_size_ms=FRAME_SIZE_MS,
            hop_size_ms=inference_hop_ms,
            sample_rate=sr
        )

        print(f"Extracted {len(frame_hcqts)} frames from padded audio")

        windows = self.create_prediction_windows(frame_hcqts)
        raw_predictions = self.batch_inference(windows)
        aggregated_predictions = self.aggregate_overlapping_predictions(
            raw_predictions, num_cqt_frames, inference_hop_ms, pad_samples
        )
        results = self.format_results_with_details(aggregated_predictions, verbose)

        if output_file:
            self.save_results(results, output_file)

        return results

    def format_results_with_details(self, aggregated_predictions, verbose):
        """
        Intialises an empty holder dictionary to keep track of detections.
        Averages the overlapping frames predicitions.
        Binarises averaged predicitions per frame.
        Displays averaged predicitions per frame. If no activity, displays no instrument.
        """
        frame_results = []
        detection_counts = {inst: 0 for inst in INSTRUMENTS_LABELS}
        total_orig_frames = len(aggregated_predictions)
        frame_duration_s = CQT_HOP / SAMPLE_RATE

        print(f"Formatting {total_orig_frames} frames with detailed predictions.")

        for frame_idx, overlapping_predictions in enumerate(aggregated_predictions):
            absolute_time = frame_idx * frame_duration_s

            if not overlapping_predictions:
                avg_pred = np.zeros(NUM_INSTRUMENTS)
            else:
                avg_pred = np.mean(np.array(overlapping_predictions), axis=0)

            binary_pred = (avg_pred > self.thresholds).astype(int)
            all_predictions = {}
            detected_instruments_verbose = []

            for j, instrument in enumerate(INSTRUMENTS_LABELS):
                is_detected = bool(binary_pred[j])
                avg_confidence = float(avg_pred[j])

                individual_preds = [p[j] for p in overlapping_predictions] if overlapping_predictions else []

                all_predictions[instrument] = {
                    'detected': is_detected,
                    'avg_confidence': avg_confidence,
                    'individual_preds': individual_preds
                }

                if is_detected:
                    detection_counts[instrument] += 1
                    pred_str = f"[{', '.join(f'{p:.2f}' for p in individual_preds)}]"
                    detected_instruments_verbose.append(f"{instrument} (AVG: {avg_confidence:.3f} from {pred_str})")

            frame_results.append({
                'frame_index': frame_idx,
                'time_ms': int(absolute_time * 1000),
                'time_seconds': float(absolute_time),
                'detected_instruments': [inst for inst, d in all_predictions.items() if d['detected']],
                'predictions': all_predictions,
            })

            if verbose:
                detected_str = ', '.join(detected_instruments_verbose) if detected_instruments_verbose else "No instruments"
                print(f"Frame {frame_idx + 1:5d} [{absolute_time:8.3f}s / {int(absolute_time * 1000):6d}ms]: {detected_str}")

        summary = self.create_summary(detection_counts, total_orig_frames)
        self.print_summary(summary)

        return {'summary': summary, 'frames': frame_results}

    def create_summary(self, detection_counts, total_frames):
        """
        Creates a dictionary summarizing inference results.
        """
        total_detections = sum(detection_counts.values())

        return {
            'total_frames': total_frames,
            'total_detections': total_detections,
            'instrument_counts': detection_counts,
            'detection_rate': total_detections / total_frames if total_frames else 0,
            'frame_duration_ms': (CQT_HOP / SAMPLE_RATE) * 1000,
            'thresholds_used': self.thresholds.tolist()
        }

    def print_summary(self, summary):
        """
        Prints a summary of the inference results.
        """
        print(f"\n=" * 80)
        print(f"Inference summary:")
        print(f"=" * 80)
        print(f"Total frames: {summary['total_frames']}")
        print(f"Frame duration: {summary['frame_duration_ms']:.2f}ms")
        print(f"Detection rate: {summary['detection_rate']:.2%}")
        print(f"Instrument frequency:")

        for instrument, count in summary['instrument_counts'].items():
            percentage = (count / summary['total_frames']) * 100 if summary['total_frames'] else 0
            print(f"  {instrument:>8}: {count:6d} frames ({percentage:5.1f}%)")

        print(f"=" * 80)

    def save_results(self, results, output_file):
        """
        Saves results to a CSV file.
        """
        try:
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            print(f"\nResults saved to: {output_file}")

        except Exception as e:
            print(f"Warning: Could not save results to {output_file}: {e}")

def main():
    try:
        device = torch.device(CONFIG['device']) if CONFIG['device'] else None
        detector = ActivityDetector(CONFIG['model_path'], device)
        results = detector.process_audio(
            audio_path=CONFIG['audio_path'],
            output_file=CONFIG['output_file'],
            verbose=CONFIG['verbose']
        )
        print("\nActivity detection completed successfully!")
        return results

    except Exception as e:
        print(f"\nError during processing: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    results = main()