"""
Scores the first onset of each instrument against the ground truth onset files of the validation set.
"""
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from instrument_detection.activity.activity_single import ActivityDetector, CONFIG as APP_CONFIG
from instrument_detection.onset.onset_detector import InstrumentAnalysisOrchestrator, EVAL_PARAMS
from instrument_detection.config import INSTRUMENTS_LABELS, TRAIN_AUDIO_DIR, TRAIN_LABEL_DIR, CACHE_DIR

# ONSET_TOLERANCE_MS is how far a predicted onset may be from a true onset and still count as correct.
# VALIDATION_CACHE_DIR holds the metadata files that name the validation clips.
EVAL_CONFIG = {
    'ONSET_TOLERANCE_MS': 100,
    'VALIDATION_CACHE_DIR': os.path.join(CACHE_DIR, 'val')
}

class PostProcessingEvaluator:
    """
    Runs the model once on the validation clips, then scores the onset post processing on the result.
    """
    def __init__(self, model_path, device=None):
        """
        Loads the activity detector, builds the onset orchestrator and starts an empty timeline cache.
        """
        self.activity_detector = ActivityDetector(model_path, device=device)
        self.orchestrator = InstrumentAnalysisOrchestrator(self.activity_detector)
        self.timelines_cache = {}
        print("Evaluator initialized successfully.")

    def _get_validation_set_files(self):
        """
        Returns the sorted wav file names of the validation clips.
        Reads the names from the metadata json files in the validation cache folder.
        Drops the last underscore part of each name, which is not part of the audio file name.
        """
        val_cache_dir = EVAL_CONFIG['VALIDATION_CACHE_DIR']

        file_basenames = set()
        for filename in os.listdir(val_cache_dir):
            if filename.startswith('metadata_') and filename.endswith('.json'):
                parts = filename.replace('metadata_', '').replace('.json', '').split('_')
                base_name = '_'.join(parts[:-1])
                file_basenames.add(base_name + '.wav')

        return sorted(list(file_basenames))

    def precompute_all_probability_timelines(self):
        """
        Runs the model on every validation clip and stores its probability timeline in the cache.
        Stores None for a clip that gives no frames.
        Doing this once lets the post processing parameters change without running the model again.
        """
        validation_files = self._get_validation_set_files()

        for filename in tqdm(validation_files, desc="Generating Probabilities"):
            audio_path = os.path.join(TRAIN_AUDIO_DIR, filename)

            activity_results = self.activity_detector.process_audio(audio_path, verbose=False)
            if not activity_results or not activity_results.get('frames'):
                self.timelines_cache[filename] = None
                continue

            frames = activity_results['frames']
            prob_timeline = np.zeros((len(frames), len(INSTRUMENTS_LABELS)))
            for i, frame_data in enumerate(frames):
                for j, instrument in enumerate(INSTRUMENTS_LABELS):
                    prob_timeline[i, j] = frame_data['predictions'][instrument]['avg_confidence']

            clip_duration = frames[-1]['time_seconds'] if frames else 0.0
            self.timelines_cache[filename] = {
                'timeline': prob_timeline,
                'duration': clip_duration,
                'thresholds': self.activity_detector.thresholds
            }

    def run_evaluation(self):
        """
        Finds the first onsets for every cached clip and prints the metrics.
        Uses the parameters in EVAL_PARAMS.
        """
        all_predictions = {}
        for filename, cached_data in tqdm(self.timelines_cache.items(), desc="Post-Processing"):
            if cached_data is None:
                all_predictions[filename] = {}
                continue

            predicted_onsets = self.orchestrator.analyze_from_timeline(
                prob_timelines=cached_data['timeline'],
                thresholds=cached_data['thresholds'],
                clip_duration=cached_data['duration'],
                params_per_instrument=EVAL_PARAMS,
                default_params=EVAL_PARAMS['default']
            )
            all_predictions[filename] = predicted_onsets if predicted_onsets else {}

        print("\nCalculating Onset Detection Metrics")
        self._calculate_and_print_metrics(all_predictions)

    def _calculate_and_print_metrics(self, all_predictions):
        """
        Compares the predicted onsets with the ground truth and prints a table for each instrument.
        For each clip, reads the true onset times from its csv file.
        Counts a prediction as a true positive if it is within the tolerance of any true onset.
        Counts a prediction as a false positive if it matches none, or if the csv file is missing.
        Counts each true onset not matched by the prediction as a false negative.
        Computes precision, recall and F1 from the counts.
        The last row uses the counts summed over all instruments.
        """
        tolerance_s = EVAL_CONFIG['ONSET_TOLERANCE_MS'] / 1000.0
        stats = {inst: {'TP': 0, 'FP': 0, 'FN': 0} for inst in INSTRUMENTS_LABELS}

        for filename, predicted_onsets in all_predictions.items():
            gt_basename = filename.replace('.wav', '')
            gt_path = os.path.join(TRAIN_LABEL_DIR, f"converted_{gt_basename}_onsets.csv")

            if not os.path.exists(gt_path):
                for instrument, onset_time in predicted_onsets.items():
                    stats[instrument]['FP'] += 1
                continue

            gt_df = pd.read_csv(gt_path)
            time_column = None
            possible_time_columns = ['start_seconds', 'onset_time_seconds', 'onset_seconds', 'onset_time', 'start_time']

            for name in possible_time_columns:
                if name in gt_df.columns:
                    time_column = name
                    break

            ground_truth = {inst: [] for inst in INSTRUMENTS_LABELS}
            for _, row in gt_df.iterrows():
                if row['instrument_name'] in ground_truth:
                    ground_truth[row['instrument_name']].append(row[time_column])

            for instrument in INSTRUMENTS_LABELS:
                pred_onset = predicted_onsets.get(instrument)
                gt_onsets = ground_truth.get(instrument, [])

                if pred_onset is not None:
                    is_tp = any(abs(pred_onset - gt_onset) <= tolerance_s for gt_onset in gt_onsets)
                    if is_tp:
                        stats[instrument]['TP'] += 1
                    else:
                        stats[instrument]['FP'] += 1

                if gt_onsets and not any(
                        abs(predicted_onsets.get(instrument, float('inf')) - gt) <= tolerance_s for gt in gt_onsets):
                    unmatched_gts = [gt for gt in gt_onsets if
                                     abs(predicted_onsets.get(instrument, float('inf')) - gt) > tolerance_s]
                    stats[instrument]['FN'] += len(unmatched_gts)

        print(f"\nOnset Detection Metrics (Tolerance: {EVAL_CONFIG['ONSET_TOLERANCE_MS']} ms)")
        print("-" * 80)
        print(
            f"{'Instrument':<15} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'TP':>4} | {'FP':>4} | {'FN':>4}")
        print("-" * 80)

        total_tp, total_fp, total_fn = 0, 0, 0

        for instrument in INSTRUMENTS_LABELS:
            tp, fp, fn = stats[instrument]['TP'], stats[instrument]['FP'], stats[instrument]['FN']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            total_tp += tp
            total_fp += fp
            total_fn += fn
            print(f"{instrument:<15} | {precision:10.3f} | {recall:10.3f} | {f1:10.3f} | {tp:4} | {fp:4} | {fn:4}")

        macro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        macro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0

        print("-" * 80)
        print(
            f"{'MACRO AVERAGE':<15} | {macro_precision:10.3f} | {macro_recall:10.3f} | {macro_f1:10.3f} | {total_tp:4} | {total_fp:4} | {total_fn:4}")
        print("=" * 80)

def main():
    """
    Evaluates the onset detection on the validation set with the model set in the activity CONFIG.
    """
    model_path = APP_CONFIG['model_path']
    device = APP_CONFIG['device']

    evaluator = PostProcessingEvaluator(model_path, device=device)
    evaluator.precompute_all_probability_timelines()
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()