import numpy as np
from activity_single import ActivityDetector, CONFIG as APP_CONFIG
from activity_gradient import FluxPeriodFinder
from activity_period import ActivityPeriodFinder
from config import INSTRUMENTS_LABELS

EVAL_PARAMS = {
    'Piano': {
        'LAG_FRAMES': 2,
        'FLUX_THRESHOLD': 0.16,
        'MIN_FLUX_PERIOD_FRAMES': 2,
        'MIN_ACTIVITY_MS': 150,
        'MIN_SILENCE_MS_TO_FILL': 500,
        'MIN_SIGNIFICANCE_SCORE': 0.02,
    },
    'Violin': {
        'LAG_FRAMES': 2,
        'FLUX_THRESHOLD': 0.09,
        'MIN_FLUX_PERIOD_FRAMES': 2,
        'MIN_ACTIVITY_MS': 150,
        'MIN_SILENCE_MS_TO_FILL': 400,
        'MIN_SIGNIFICANCE_SCORE': 0.01,
    },
    'Cello': {
        'LAG_FRAMES': 2,
        'FLUX_THRESHOLD': 0.16,
        'MIN_FLUX_PERIOD_FRAMES': 2,
        'MIN_ACTIVITY_MS': 150,
        'MIN_SILENCE_MS_TO_FILL': 400,
        'MIN_SIGNIFICANCE_SCORE': 0.03,
    },
    'Viola': {
        'LAG_FRAMES': 2,
        'FLUX_THRESHOLD': 0.16,
        'MIN_FLUX_PERIOD_FRAMES': 2,
        'MIN_ACTIVITY_MS': 150,
        'MIN_SILENCE_MS_TO_FILL': 400,
        'MIN_SIGNIFICANCE_SCORE': 0.03,
    },
    'Contrabass': {
        'LAG_FRAMES': 2,
        'FLUX_THRESHOLD': 0.05,
        'MIN_FLUX_PERIOD_FRAMES': 2,
        'MIN_ACTIVITY_MS': 150,
        'MIN_SILENCE_MS_TO_FILL': 700,
        'MIN_SIGNIFICANCE_SCORE': 0.01,
    },
    'Clarinet': {
        'LAG_FRAMES': 2,
        'FLUX_THRESHOLD': 0.08,
        'MIN_FLUX_PERIOD_FRAMES': 2,
        'MIN_ACTIVITY_MS': 150,
        'MIN_SILENCE_MS_TO_FILL': 600,
        'MIN_SIGNIFICANCE_SCORE': 0.015,
    },
    'Bassoon': {
        'LAG_FRAMES': 2,
        'FLUX_THRESHOLD': 0.08,
        'MIN_FLUX_PERIOD_FRAMES': 2,
        'MIN_ACTIVITY_MS': 150,
        'MIN_SILENCE_MS_TO_FILL': 700,
        'MIN_SIGNIFICANCE_SCORE': 0.05,
    },
    'Oboe': {
        'LAG_FRAMES': 2,
        'FLUX_THRESHOLD': 0.08,
        'MIN_FLUX_PERIOD_FRAMES': 2,
        'MIN_ACTIVITY_MS': 150,
        'MIN_SILENCE_MS_TO_FILL': 500,
        'MIN_SIGNIFICANCE_SCORE': 0.015,
    },
    'Flute': {
        'LAG_FRAMES': 2,
        'FLUX_THRESHOLD': 0.07,
        'MIN_FLUX_PERIOD_FRAMES': 2,
        'MIN_ACTIVITY_MS': 150,
        'MIN_SILENCE_MS_TO_FILL': 700,
        'MIN_SIGNIFICANCE_SCORE': 0.11,
    },
    'French Horn': {
        'LAG_FRAMES': 2,
        'FLUX_THRESHOLD': 0.08,
        'MIN_FLUX_PERIOD_FRAMES': 2,
        'MIN_ACTIVITY_MS': 150,
        'MIN_SILENCE_MS_TO_FILL': 500,
        'MIN_SIGNIFICANCE_SCORE': 0.01,
    },
    'Trumpet': {
        'LAG_FRAMES': 2,
        'FLUX_THRESHOLD': 0.04,
        'MIN_FLUX_PERIOD_FRAMES': 2,
        'MIN_ACTIVITY_MS': 50,
        'MIN_SILENCE_MS_TO_FILL': 800,
        'MIN_SIGNIFICANCE_SCORE': 0.006,
    },
    'default': {
        'LAG_FRAMES': 2,
        'FLUX_THRESHOLD': 0.16,
        'MIN_FLUX_PERIOD_FRAMES': 2,
        'MIN_ACTIVITY_MS': 150,
        'MIN_SILENCE_MS_TO_FILL': 500,
        'MIN_SIGNIFICANCE_SCORE': 0.02,
    }
}

class InstrumentAnalysisOrchestrator:
    def __init__(self, activity_detector: ActivityDetector):
        self.activity_detector = activity_detector
        self.activity_finder = ActivityPeriodFinder()
        self.flux_finder = FluxPeriodFinder()

    def analyze(self, audio_path: str, params_per_instrument: dict, default_params: dict):
        activity_results = self.activity_detector.process_audio(audio_path, verbose=False)
        if not activity_results or not activity_results.get('frames'):
            return None

        frames = activity_results['frames']
        prob_timelines = np.zeros((len(frames), len(INSTRUMENTS_LABELS)))
        for i, frame_data in enumerate(frames):
            for j, instrument in enumerate(INSTRUMENTS_LABELS):
                prob_timelines[i, j] = frame_data['predictions'][instrument]['avg_confidence']

        thresholds = self.activity_detector.thresholds
        clip_duration = frames[-1]['time_seconds'] if frames else 0.0

        return self.analyze_from_timeline(
            prob_timelines, thresholds, clip_duration, params_per_instrument, default_params
        )

    def analyze_from_timeline(self, prob_timelines, thresholds, clip_duration, params_per_instrument, default_params):
        activity_periods = self.activity_finder.find_periods(
            prob_timelines, thresholds, params_per_instrument, default_params
        )
        flux_periods, _ = self.flux_finder.find_periods(
            prob_timelines, params_per_instrument, default_params
        )

        first_onsets = self._find_first_significant_onset(
            activity_periods, flux_periods, clip_duration, params_per_instrument, default_params
        )
        return first_onsets

    def _find_first_significant_onset(self, activity_periods, flux_periods, clip_duration, params_per_instrument,
                                      default_params):
        first_onsets = {}
        for instrument in INSTRUMENTS_LABELS:
            activity_list = activity_periods.get(instrument, [])
            flux_list = flux_periods.get(instrument, [])

            if not activity_list or not flux_list:
                continue

            inst_params = params_per_instrument.get(instrument, default_params)
            min_significance_score = inst_params['MIN_SIGNIFICANCE_SCORE']

            candidate_onsets = []

            for activity_start, activity_end, avg_prob in activity_list:
                period_duration = activity_end - activity_start
                duration_factor = period_duration / clip_duration if clip_duration > 0 else 0
                significance_score = avg_prob * duration_factor

                if significance_score >= min_significance_score:
                    for flux_start, flux_end, peak_flux_time in flux_list:
                        if max(flux_start, activity_start) <= min(flux_end, activity_end):
                            candidate_onsets.append(peak_flux_time)
                            break

            if candidate_onsets:
                first_onsets[instrument] = min(candidate_onsets)

        return first_onsets

def main():
    try:
        model_path = APP_CONFIG['model_path']
        audio_path = APP_CONFIG['audio_path']
        device = APP_CONFIG['device']

        activity_detector = ActivityDetector(model_path, device=device)
        orchestrator = InstrumentAnalysisOrchestrator(activity_detector)

        onsets = orchestrator.analyze(
            audio_path,
            params_per_instrument=EVAL_PARAMS,
            default_params=EVAL_PARAMS['default']
        )

        print("\n\n" + "=" * 80)
        print("FINAL CONFIRMED FIRST ONSETS")
        print("=" * 80)

        for instrument in sorted(onsets.keys()):
            onset_time = onsets[instrument]
            print(f"  {instrument:<15}: {onset_time:.3f}s")
        print("=" * 80)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()