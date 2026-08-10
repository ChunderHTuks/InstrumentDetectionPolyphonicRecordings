import numpy as np
import os
from activity_single import ActivityDetector, CONFIG as APP_CONFIG
from config import INSTRUMENTS_LABELS, CQT_HOP, SAMPLE_RATE

CONFIG = {
    'MIN_ACTIVITY_MS': 150,
    'MIN_SILENCE_MS_TO_FILL': 600,
    'MIN_SIGNIFICANCE_SCORE': 0.08
}

class ActivityPeriodFinder:
    def __init__(self, activity_detector: ActivityDetector):
        if not isinstance(activity_detector, ActivityDetector):
            raise TypeError("activity_detector must be an instance of ActivityDetector")

        self.activity_detector = activity_detector

    def filter_state_timeline(self, timeline, min_duration_frames, target_state, new_state):
        """
        Adds a 0 or 1 to the start and end of the timeline.
        Uses the diffs method to find transitions.
        Extracts the indices of period starts and period ends.
        Checks if the duration between start and end is less than min_duration_frames.
        If not, marks active periods.
        """
        filtered_timeline = timeline.copy()
        padded = np.pad(timeline, (1, 1), 'constant', constant_values=(1 - target_state, 1 - target_state))
        diffs = np.diff(padded)

        starts = np.where(diffs == target_state - (1 - target_state))[0]
        ends = np.where(diffs == (1 - target_state) - target_state)[0]

        for start, end in zip(starts, ends):
            duration = end - start
            if duration < min_duration_frames and start > 0:
                filtered_timeline[start:end] = new_state

        return filtered_timeline

    def find_periods(self, audio_path: str) -> dict:
        """
        Extracts config parameters.
        Parses the audio through the activity detector.
        Finds the number of frames corresponding to the min activity and the min silence.
        Binarises the instrument timelines, and 0s the first frame to avoid false positives.
        Fills in periods shorter than min_silence frames.
        Removes periods shorter than min_activity_frames.
        Calculates the average activation probability per activity segment.
        Does significance scoring by the product of (segment duration / song duration) * avg probability.
        Filters periods by average significance score.
        """
        min_activity_ms = CONFIG['MIN_ACTIVITY_MS']
        min_silence_ms = CONFIG['MIN_SILENCE_MS_TO_FILL']
        min_score_threshold = CONFIG['MIN_SIGNIFICANCE_SCORE']
        instrument_thresholds = self.activity_detector.thresholds

        print("\n" + "=" * 80)
        print(f"Finding significant activity periods for: {os.path.basename(audio_path)}")
        print("=" * 80)

        activity_results = self.activity_detector.process_audio(audio_path, verbose=False)

        frames = activity_results['frames']
        prob_timelines = np.zeros((len(frames), len(INSTRUMENTS_LABELS)))
        for i, frame_data in enumerate(frames):
            for j, instrument in enumerate(INSTRUMENTS_LABELS):
                prob_timelines[i, j] = frame_data['predictions'][instrument]['avg_confidence']

        clip_duration = frames[-1]['time_seconds'] if frames else 0.0

        frame_duration_s = CQT_HOP / SAMPLE_RATE
        min_activity_frames = int(min_activity_ms / (1000 * frame_duration_s))
        min_silence_frames = int(min_silence_ms / (1000 * frame_duration_s))

        final_filtered_periods = {}

        for i, instrument in enumerate(INSTRUMENTS_LABELS):
            p_t = prob_timelines[:, i]
            threshold = instrument_thresholds[i]

            s_raw = (p_t >= threshold).astype(int)
            if len(s_raw) > 0:
                s_raw[0] = 0

            s_filled = self.filter_state_timeline(s_raw, min_silence_frames, target_state=0, new_state=1)
            s_final = self.filter_state_timeline(s_filled, min_activity_frames, target_state=1, new_state=0)

            diffs = np.diff(s_final, prepend=0, append=0)
            start_indices = np.where(diffs == 1)[0]
            end_indices = np.where(diffs == -1)[0]

            significant_periods = []
            for start_idx, end_idx in zip(start_indices, end_indices):
                start_time = start_idx * frame_duration_s
                end_time = (end_idx - 1) * frame_duration_s
                period_probabilities = p_t[start_idx:end_idx]
                avg_prob = np.mean(period_probabilities) if len(period_probabilities) > 0 else 0

                period_duration = end_time - start_time
                duration_factor = period_duration / clip_duration if clip_duration > 0 else 0
                significance_score = avg_prob * duration_factor

                if significance_score >= min_score_threshold:
                    significant_periods.append((start_time, end_time, avg_prob))

            final_filtered_periods[instrument] = significant_periods

        return final_filtered_periods

def main():
    try:
        model_path = APP_CONFIG['model_path']
        audio_path = APP_CONFIG['audio_path']
        device = APP_CONFIG['device']

        activity_detector = ActivityDetector(model_path, device=device)
        period_finder = ActivityPeriodFinder(activity_detector)

        periods = period_finder.find_periods(audio_path=audio_path)

        print("Thresholds Used:")
        for i, instrument in enumerate(INSTRUMENTS_LABELS):
            print(f"  - {instrument:<15}: {activity_detector.thresholds[i]:.3f}")
        print("-" * 80)

        for instrument, period_list in periods.items():
            if period_list:
                print(f"{instrument} activity periods:")
                for start_time, end_time, avg_prob in period_list:
                    print(f"  {start_time:7.3f}s - {end_time:7.3f}s (Average: {avg_prob:.2f})")
        print("=" * 80)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()