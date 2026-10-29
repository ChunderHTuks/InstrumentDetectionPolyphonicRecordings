"""
Searches a grid of onset parameters and prints the best set for each instrument on the validation set.
"""
import numpy as np
import os
import time
import pandas as pd
from tqdm import tqdm
import itertools
import multiprocessing as mp
from instrument_detection.activity.activity_single import ActivityDetector, CONFIG as APP_CONFIG
from instrument_detection.onset.onset_detector import InstrumentAnalysisOrchestrator
from instrument_detection.config import INSTRUMENTS_LABELS, CACHE_DIR, VAL_AUDIO_DIR, VAL_LABEL_DIR

# Every combination of these values is tried. Each key is described in activity_gradient.py
# or activity_period.py.
PARAM_GRID = {
    'LAG_FRAMES': [1, 2, 3, 4, 5, 6, 7],
    'FLUX_THRESHOLD': [0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18],
    'MIN_FLUX_PERIOD_FRAMES': [1, 2, 3, 4, 5, 6, 7],
    'MIN_ACTIVITY_MS': [50, 100, 150, 250, 300, 350, 400, 450, 500],
    'MIN_SILENCE_MS_TO_FILL': [300, 400, 500, 600, 700, 800, 900, 1000],
    'MIN_SIGNIFICANCE_SCORE': [0.008, 0.009, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12]
}

# ONSET_TOLERANCE_MS is how far a predicted onset may be from a true onset and still count as correct.
# VERBOSE_LOGGING prints a line each time an instrument gets a new best score.
EVAL_CONFIG = {
    'ONSET_TOLERANCE_MS': 100,
    'VERBOSE_LOGGING': False
}

# Holds the orchestrator and the cached timelines inside each worker process.
worker_data = {}

def init_worker(orchestrator_instance, timelines_cache_data):
    """
    Stores the orchestrator and the timeline cache once per worker process.
    This avoids sending them again with every parameter combination.
    """
    worker_data['orchestrator'] = orchestrator_instance
    worker_data['timelines_cache'] = timelines_cache_data


def process_combo(combo_tuple):
    """
    Scores one parameter combination and returns the parameters with the F1 score of each instrument.
    Uses the same parameters for every instrument.
    Finds the first onsets of every cached clip with these parameters.
    Counts true positives, false positives and false negatives against the ground truth csv files.
    A clip without a csv file adds a false positive for each predicted onset.
    Turns the counts into an F1 score for each instrument.
    """
    param_names, combo = combo_tuple
    current_params = dict(zip(param_names, combo))
    temp_eval_params = {inst: current_params for inst in INSTRUMENTS_LABELS}
    temp_eval_params['default'] = current_params

    all_predictions = {}
    for filename, cached_data in worker_data['timelines_cache'].items():
        if cached_data is None:
            all_predictions[filename] = {}
            continue

        predicted_onsets = worker_data['orchestrator'].analyze_from_timeline(
            prob_timelines=cached_data['timeline'],
            thresholds=cached_data['thresholds'],
            clip_duration=cached_data['duration'],
            params_per_instrument=temp_eval_params,
            default_params=temp_eval_params['default']
        )
        all_predictions[filename] = predicted_onsets if predicted_onsets else {}

    tolerance_s = EVAL_CONFIG['ONSET_TOLERANCE_MS'] / 1000.0
    stats = {inst: {'TP': 0, 'FP': 0, 'FN': 0} for inst in INSTRUMENTS_LABELS}
    for filename, predicted_onsets in all_predictions.items():
        gt_basename = filename.replace('.wav', '')
        gt_path = os.path.join(VAL_LABEL_DIR, f"converted_{gt_basename}_onsets.csv")

        if not os.path.exists(gt_path):
            for instrument, onset_time in predicted_onsets.items(): stats[instrument]['FP'] += 1
            continue

        gt_df = pd.read_csv(gt_path)
        time_column = 'onset_time_seconds' if 'onset_time_seconds' in gt_df.columns else gt_df.columns[1]

        ground_truth = {inst: [] for inst in INSTRUMENTS_LABELS}
        for _, row in gt_df.iterrows():
            if row['instrument_name'] in ground_truth: ground_truth[row['instrument_name']].append(row[time_column])

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
                unmatched_gts = [gt for gt in gt_onsets if abs(predicted_onsets.get(instrument, float('inf')) - gt) > tolerance_s]
                stats[instrument]['FN'] += len(unmatched_gts)

    f1_scores = {}
    for instrument in INSTRUMENTS_LABELS:
        tp, fp, fn = stats[instrument]['TP'], stats[instrument]['FP'], stats[instrument]['FN']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores[instrument] = f1

    return current_params, f1_scores


class ParameterTuner:
    """
    Computes the validation probabilities once, then searches the parameter grid on them.
    """
    def __init__(self, model_path, device=None):
        """
        Loads the activity detector, builds the onset orchestrator and starts an empty timeline cache.
        """
        self.activity_detector = ActivityDetector(model_path, device=device)
        self.orchestrator = InstrumentAnalysisOrchestrator(self.activity_detector)
        self.timelines_cache = {}
        print("Tuner initialized successfully.")

    def _get_validation_set_files(self):
        """
        Returns the sorted wav file names of the validation clips.
        Reads the names from the metadata json files in the validation cache folder.
        Drops the last underscore part of each name, which is not part of the audio file name.
        """
        val_cache_dir = os.path.join(CACHE_DIR, 'val')

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
        The model runs only once so each parameter combination only repeats the cheap post processing.
        """
        validation_files = self._get_validation_set_files()

        for filename in tqdm(validation_files, desc="Generating Probabilities"):
            audio_path = os.path.join(VAL_AUDIO_DIR, filename)
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
                'timeline': prob_timeline, 'duration': clip_duration, 'thresholds': self.activity_detector.thresholds
            }

    def run_parallel_grid_search(self):
        """
        Tries every combination in PARAM_GRID and keeps the best parameters for each instrument.
        Spreads the combinations over all CPU cores except one.
        Replaces the best entry of an instrument whenever a combination gives a higher F1 score.
        Prints the best parameters at the end.
        """
        param_names = list(PARAM_GRID.keys())
        param_values = list(PARAM_GRID.values())
        combinations = list(itertools.product(*param_values))
        total_combinations = len(combinations)

        job_args = [(param_names, combo) for combo in combinations]

        num_workers = max(1, mp.cpu_count() - 1)
        print(f"Distributing {total_combinations} jobs across {num_workers} CPU cores.")

        best_f1_scores = {inst: -1.0 for inst in INSTRUMENTS_LABELS}
        best_params = {inst: {} for inst in INSTRUMENTS_LABELS}

        with mp.Pool(processes=num_workers, initializer=init_worker,
                     initargs=(self.orchestrator, self.timelines_cache)) as pool:
            results_iterator = pool.imap_unordered(process_combo, job_args)
            progress_bar = tqdm(results_iterator, total=total_combinations, desc="Parallel Grid Search")

            for current_params, current_f1_scores in progress_bar:
                for instrument in INSTRUMENTS_LABELS:
                    if current_f1_scores[instrument] > best_f1_scores[instrument]:
                        if EVAL_CONFIG['VERBOSE_LOGGING']:
                            progress_bar.write(
                                f"New best for {instrument:<15}: {current_f1_scores[instrument]:.4f}")
                        best_f1_scores[instrument] = current_f1_scores[instrument]
                        best_params[instrument] = current_params.copy()

        self._print_final_results(best_params, best_f1_scores)

    def _print_final_results(self, best_params, best_f1_scores):
        """
        Prints the best parameters as an EVAL_PARAMS dict that can be pasted into onset_detector.py.
        """
        print("EVAL_PARAMS = {")
        for instrument in INSTRUMENTS_LABELS:
            print(f"    '{instrument}': {{")
            params = best_params.get(instrument)
            if not params:
                params = list(PARAM_GRID.values())[0]
            for key, value in params.items():
                print(f"        '{key}': {value},")


def main():
    """
    Runs the grid search on the validation set and prints how long it took.
    """
    model_path = APP_CONFIG['model_path']
    device = APP_CONFIG['device']

    tuner = ParameterTuner(model_path, device=device)
    tuner.precompute_all_probability_timelines()

    start_time = time.time()
    tuner.run_parallel_grid_search()
    end_time = time.time()

    print(f"\nTotal parallel grid search took: {(end_time - start_time) / 60:.2f} minutes.")

if __name__ == "__main__":
    main()