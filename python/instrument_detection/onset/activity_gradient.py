"""
Finds the periods where an instrument probability rises sharply, called flux periods.
This is one half of the onset detection, the other half is activity_period.py.
"""
import numpy as np
import os
from instrument_detection.activity.activity_single import ActivityDetector, CONFIG as APP_CONFIG
from instrument_detection.config import INSTRUMENTS_LABELS, CQT_HOP, SAMPLE_RATE
import traceback

# LAG_FRAMES is how far back the flux looks, FLUX_THRESHOLD is the minimum rise that counts as high flux.
# MIN_FLUX_PERIOD_FRAMES drops high flux runs that are too short to be a real entry.
# The two VERBOSE entries only control the frame by frame printout.
CONFIG = {
    'VERBOSE_FRAME_BY_FRAME': True,
    'LAG_FRAMES': 2,
    'FLUX_THRESHOLD': 0.07,
    'MIN_FLUX_PERIOD_FRAMES': 2,
    'VERBOSE_PRINT_THRESHOLD': 0.11,
}

class FluxPeriodFinder:
    """
    Turns the model probabilities into periods of rapid probability increase for each instrument.
    """
    def __init__(self, activity_detector: ActivityDetector):
        """
        Checks that it was given an ActivityDetector and keeps it for running inference later.
        """
        if not isinstance(activity_detector, ActivityDetector):
            raise TypeError("activity_detector must be an instance of ActivityDetector")

        self.activity_detector = activity_detector

    def filter_short_periods(self, timeline, min_duration_frames):
        """
        Adds a 0 to the start and end of the timeline.
        Uses the diffs method to find transitions.
        Extracts the indices of period starts and period ends.
        Checks if the duration between start and end is less than min_duration_frames.
        If not, marks active periods.
        """
        filtered_timeline = timeline.copy()
        padded = np.pad(timeline, (1, 1), 'constant', constant_values=0)
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        for start, end in zip(starts, ends):
            duration = end - start
            if duration < min_duration_frames:
                filtered_timeline[start:end] = 0

        return filtered_timeline

    def determine_flux_periods(self, audio_path: str) -> dict:
        """
        Extracts config parameters.
        Parses the audio through the activity detector.
        Adds lag frames worth of 0s at the beginning of the timeline and removes lag frames worth of values at the end.
        Calculates the difference.
        Finds the periods which are greater than the flux threshold, and filters periods that aren't of desired length.
        Finds the peak flux region.
        """
        verbose = CONFIG['VERBOSE_FRAME_BY_FRAME']
        lag_frames = CONFIG['LAG_FRAMES']
        flux_threshold = CONFIG['FLUX_THRESHOLD']
        min_flux_period_frames = CONFIG['MIN_FLUX_PERIOD_FRAMES']
        verbose_print_threshold = CONFIG['VERBOSE_PRINT_THRESHOLD']

        activity_results = self.activity_detector.process_audio(audio_path, verbose=False)
        frames = activity_results['frames']
        prob_timelines = np.zeros((len(frames), len(INSTRUMENTS_LABELS)))

        for i, frame_data in enumerate(frames):
            for j, instrument in enumerate(INSTRUMENTS_LABELS):
                prob_timelines[i, j] = frame_data['predictions'][instrument]['avg_confidence']

        frame_duration_s = CQT_HOP / SAMPLE_RATE
        p_t_lagged = np.pad(prob_timelines, ((lag_frames, 0), (0, 0)), 'constant')[:-lag_frames, :]
        rectified_flux = np.maximum(0, prob_timelines - p_t_lagged)

        if verbose:
            for frame_idx in range(len(prob_timelines)):
                absolute_time = frame_idx * frame_duration_s
                flux_strings = []
                for inst_idx, instrument in enumerate(INSTRUMENTS_LABELS):
                    flux_value = rectified_flux[frame_idx, inst_idx]
                    if flux_value > verbose_print_threshold:
                        flux_strings.append(f"{instrument} (Flux: {flux_value:+.3f})")

                detected_str = ', '.join(flux_strings) if flux_strings else "No significant flux"
                print(f"Frame {frame_idx + 1:5d} [{absolute_time:8.3f}s]: {detected_str}")
            print("-" * 50)

        final_periods = {}
        for i, instrument in enumerate(INSTRUMENTS_LABELS):
            instrument_flux = rectified_flux[:, i]
            high_flux_state = (instrument_flux > flux_threshold).astype(int)
            filtered_high_flux_state = self.filter_short_periods(high_flux_state, min_flux_period_frames)

            diffs = np.diff(filtered_high_flux_state, prepend=0, append=0)
            start_indices = np.where(diffs == 1)[0]
            end_indices = np.where(diffs == -1)[0]

            instrument_periods = []
            for start_idx, end_idx in zip(start_indices, end_indices):
                period_flux_values = instrument_flux[start_idx:end_idx]
                peak_local_idx = np.argmax(period_flux_values)
                peak_global_idx = start_idx + peak_local_idx

                start_time = start_idx * frame_duration_s
                end_time = (end_idx - 1) * frame_duration_s

                peak_flux_time = peak_global_idx * frame_duration_s
                instrument_periods.append((start_time, end_time, peak_flux_time))

            final_periods[instrument] = instrument_periods

        return final_periods


def main():
    """
    Finds and prints the flux periods of the audio file set in the activity detector CONFIG.
    """
    try:
        model_path = APP_CONFIG['model_path']
        audio_path = APP_CONFIG['audio_path']
        device = APP_CONFIG['device']

        activity_detector = ActivityDetector(model_path, device=device)
        flux_period_finder = FluxPeriodFinder(activity_detector)
        periods = flux_period_finder.determine_flux_periods(audio_path=audio_path)

        print("\n\n" + "=" * 80)
        print("INSTRUMENT FLUX PERIODS")
        print("=" * 80)

        for instrument, period_list in periods.items():
            if period_list:
                print(f"{instrument} high flux periods:")
                for start_time, end_time, peak_time in period_list:
                    print(f"  Period: {start_time:7.3f}s - {end_time:7.3f}s  |  Peak Flux at: {peak_time:7.3f}s")
        print("=" * 80)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()