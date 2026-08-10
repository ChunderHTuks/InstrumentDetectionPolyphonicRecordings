import torch
import numpy as np
import json
from datetime import datetime
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from model import build_model
from hcqt_preprocessing import InstrumentDataset
from config import *

CONFIG = {
    'model_path': './models/instrument_model_2025-10-16_13-42-05.pth',
    'output_file': None,
    'verbose': True
}

class ModelTester:
    def __init__(self, config=None):
        if config is None:
            config = CONFIG

        self.config = config
        self.device = self._setup_device()
        self.model = self._load_model()
        self.thresholds = self._get_thresholds()

    def _setup_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()

        return device

    def _load_model(self):
        model_path = self.config['model_path']

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        model = build_model().to(self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

            if 'final_test_metrics' in checkpoint:
                print(f"Original test F1 score: {checkpoint['final_test_metrics']['macro']['f1']:.4f}")
            if 'dataset_hash' in checkpoint:
                print(f"Training dataset hash: {checkpoint.get('dataset_hash', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dict directly")

        model.to(self.device)
        model.eval()

        print(f"Model parameters: {model.count_parameters():,}")

        return model

    def _get_thresholds(self):
        checkpoint = torch.load(self.config['model_path'], map_location=self.device)

        if 'weights_info' in checkpoint and 'temporal_thresholds' in checkpoint['weights_info']:
            thresholds = np.array(checkpoint['weights_info']['temporal_thresholds'])
            print(f"Using thresholds from 'weights_info': {np.round(thresholds, 3)}")
            return thresholds
        elif 'thresholds' in checkpoint:
            thresholds = np.array(checkpoint['thresholds'])
            return thresholds
        else:
            thresholds = np.full(NUM_INSTRUMENTS, 0.5)
            return thresholds

    def create_test_dataset(self):

        test_dataset = InstrumentDataset(
            audio_dir=TEST_AUDIO_DIR,
            label_dir=TEST_LABEL_DIR,
            cache_dir="./cache/test",
            frame_size_ms=FRAME_SIZE_MS,
            hop_size_ms=HOP_SIZE_MS,
            n_harmonics=N_HARMONICS,
            sample_rate=SAMPLE_RATE,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE,
            hop_length=CQT_HOP,
            use_cache=USE_CACHE,
            label_smoothing=0.0,
            n_processes=1
        )
        return test_dataset

    def create_test_dataloader(self, test_dataset):
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        print(f"Test DataLoader created: {len(test_loader)} batches")
        return test_loader

    def calculate_metrics(self, predictions, targets):
        n_windows, n_instruments, n_time_frames = predictions.shape
        thresholds = self._get_thresholds()
        binary_predictions = np.zeros_like(predictions)

        for i in range(n_instruments):
            binary_predictions[:, i, :] = (predictions[:, i, :] > thresholds[i]).astype(int)

        total_frames = n_windows * n_time_frames
        pred_frames = binary_predictions.transpose(0, 2, 1).reshape(total_frames, n_instruments)
        target_frames = targets.transpose(0, 2, 1).reshape(total_frames, n_instruments)
        raw_pred_frames = predictions.transpose(0, 2, 1).reshape(total_frames, n_instruments)

        print(f"Frame-level evaluation:")
        print(f"Total frames: {total_frames:,}")
        print(f"Frame duration: 11.61ms")
        print(f"Total evaluation time: {total_frames * 11.61 / 1000:.2f}s")
        print(f"Instruments: {n_instruments}")
        print(f"Thresholds: {thresholds}")
        print(f"=" * 80)

        instrument_metrics = {}
        overall_stats = {
            'total_frames': total_frames,
            'frame_duration_ms': 11.61,
            'n_instruments': n_instruments,
            'thresholds_used': thresholds.tolist()
        }

        global_tp = np.sum((pred_frames == 1) & (target_frames == 1))
        global_tn = np.sum((pred_frames == 0) & (target_frames == 0))
        global_fp = np.sum((pred_frames == 1) & (target_frames == 0))
        global_fn = np.sum((pred_frames == 0) & (target_frames == 1))

        global_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
        global_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
        global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0
        global_accuracy = (global_tp + global_tn) / (global_tp + global_tn + global_fp + global_fn)

        overall_stats.update({
            'global_accuracy': global_accuracy,
            'global_precision': global_precision,
            'global_recall': global_recall,
            'global_f1_score': global_f1,
            'global_tp': int(global_tp),
            'global_tn': int(global_tn),
            'global_fp': int(global_fp),
            'global_fn': int(global_fn)
        })

        print(f"Accuracy:  {global_accuracy:.4f}")
        print(f"Precision: {global_precision:.4f}")
        print(f"Recall:    {global_recall:.4f}")
        print(f"F1-Score:  {global_f1:.4f}")
        print(f"TP: {global_tp:,}, TN: {global_tn:,}, FP: {global_fp:,}, FN: {global_fn:,}")
        print(f"\nPER-INSTRUMENT METRICS:")

        for i, instrument in enumerate(INSTRUMENTS_LABELS):
            pred_inst = pred_frames[:, i]
            target_inst = target_frames[:, i]
            raw_pred_inst = raw_pred_frames[:, i]

            tp = np.sum((pred_inst == 1) & (target_inst == 1))
            tn = np.sum((pred_inst == 0) & (target_inst == 0))
            fp = np.sum((pred_inst == 1) & (target_inst == 0))
            fn = np.sum((pred_inst == 0) & (target_inst == 1))

            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            positive_frames = np.sum(target_inst == 1)
            negative_frames = np.sum(target_inst == 0)
            predicted_positive = np.sum(pred_inst == 1)
            avg_confidence_positive = np.mean(raw_pred_inst[target_inst == 1]) if positive_frames > 0 else 0
            avg_confidence_negative = np.mean(raw_pred_inst[target_inst == 0]) if negative_frames > 0 else 0

            instrument_metrics[instrument] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'positive_frames': int(positive_frames),
                'negative_frames': int(negative_frames),
                'predicted_positive_frames': int(predicted_positive),
                'avg_confidence_when_present': float(avg_confidence_positive),
                'avg_confidence_when_absent': float(avg_confidence_negative),
                'threshold_used': float(thresholds[i])
            }

            print(f"\n{instrument.upper()}:")
            print(f"  Accuracy:    {accuracy:.4f}")
            print(f"  Precision:   {precision:.4f}")
            print(f"  Recall:      {recall:.4f}")
            print(f"  F1-Score:    {f1:.4f}")
            print(f"  Specificity: {specificity:.4f}")
            print(f"  TP: {tp:,}, TN: {tn:,}, FP: {fp:,}, FN: {fn:,}")
            print(f"  Ground Truth Positive: {positive_frames:,} frames ({positive_frames / total_frames * 100:.2f}%)")
            print(f"  Predicted Positive: {predicted_positive:,} frames ({predicted_positive / total_frames * 100:.2f}%)")
            print(f"  Avg Confidence (Present): {avg_confidence_positive:.4f}")
            print(f"  Avg Confidence (Absent):  {avg_confidence_negative:.4f}")

        return {
            'overall_stats': overall_stats,
            'instrument_metrics': instrument_metrics,
            'global_metrics': {
                'accuracy': global_accuracy,
                'precision': global_precision,
                'recall': global_recall,
                'f1_score': global_f1
            }
        }

    def print_metrics(self, metrics, instrument_names):
        print("\nPer-instrument Metrics:")
        print(f"{'Instrument':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
        print("-" * 80)

        for i, instrument in enumerate(instrument_names):
            precision = metrics['instrument_metrics']['precision'][i]
            recall = metrics['instrument_metrics']['recall'][i]
            f1 = metrics['instrument_metrics']['f1'][i]
            support = metrics['instrument_metrics']['support'][i]

            print(f"{instrument:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10.0f}")

        print("\nOverall Stats:")
        for key, value in metrics['overall_stats'].items():
            print(f"{key.capitalize():<20}: {value:.4f}")

        print("\nActivity Stats:")
        for key, value in metrics['activity_stats'].items():
            print(f"{key.capitalize():<20}: {value:.4f}")

        print("\nGlobal Metrics:")
        global_metrics = metrics['global_metrics']
        print(f"Accuracy         : {global_metrics['accuracy']:.4f}")
        print(f"Precision        : {global_metrics['precision']:.4f}")
        print(f"Recall           : {global_metrics['recall']:.4f}")
        print(f"F1 Score         : {global_metrics['f1_score']:.4f}")

    def evaluate_model(self, test_loader):
        self.model.eval()

        all_predictions = []
        all_targets = []
        total_batches = len(test_loader)

        print(f"Processing {total_batches} batches.")

        with torch.no_grad():
            for batch_idx, (data, target, _, _) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                with autocast():
                    output = self.model(data)

                predictions = torch.sigmoid(output).cpu().numpy()
                targets = target.cpu().numpy()

                all_predictions.append(predictions)
                all_targets.append(targets)

                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                    print(f"  Processed batch {batch_idx + 1}/{total_batches}")

        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)

        print(f"\nEvaluation completed!")
        return all_predictions, all_targets

    def run_test_evaluation(self):
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        test_dataset = self.create_test_dataset()
        test_loader = self.create_test_dataloader(test_dataset)
        predictions, targets = self.evaluate_model(test_loader)
        metrics = self.calculate_metrics(predictions, targets)

        self.print_metrics(metrics)

        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'dataset_info': {
                'total_samples': len(predictions),
                'num_instruments': len(INSTRUMENTS_LABELS),
                'instruments': INSTRUMENTS_LABELS
            },
            'metrics': metrics,
            'summary': {
                'global_f1': metrics['global_metrics']['f1_score'],
                'global_precision': metrics['global_metrics']['precision'],
                'global_recall': metrics['global_metrics']['recall'],
                'global_accuracy': metrics['global_metrics']['accuracy'],
                'frames_with_any_instrument': metrics['activity_stats']['frames_with_any_instrument'],
                'percentage_active_frames': metrics['activity_stats']['percentage_active_frames']
            }
        }

        if self.config['output_file']:
            self.save_results(results)

        return results

    def print_metrics(self, metrics):
        print("\nPer-instrument Metrics:")
        print(f"{'Instrument':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Accuracy':<10} {'Support':<10}")
        print("-" * 80)

        for instrument, stats in metrics['instrument_metrics'].items():
            precision = stats['precision']
            recall = stats['recall']
            f1 = stats['f1_score']
            accuracy = stats['accuracy']
            support = stats['positive_frames'] + stats['negative_frames']

            print(f"{instrument:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {accuracy:<10.4f} {support:<10}")

        print("\nGlobal Metrics:")
        gm = metrics['global_metrics']
        print(f"Accuracy         : {gm['accuracy']:.4f}")
        print(f"Precision        : {gm['precision']:.4f}")
        print(f"Recall           : {gm['recall']:.4f}")
        print(f"F1 Score         : {gm['f1_score']:.4f}")

        print("\nOverall Stats:")
        for k, v in metrics['overall_stats'].items():
            if isinstance(v, float):
                print(f"{k:<25}: {v:.4f}")
            else:
                print(f"{k:<25}: {v}")

        print("\nActivity Stats:")
        for k, v in metrics['activity_stats'].items():
            if isinstance(v, float):
                print(f"{k:<25}: {v:.2f}")
            else:
                print(f"{k:<25}: {v}")

    def save_results(self, results):
        output_file = self.config['output_file']
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

def main():
    tester = ModelTester(CONFIG)
    results = tester.run_test_evaluation()

    return results


if __name__ == "__main__":
    results = main()