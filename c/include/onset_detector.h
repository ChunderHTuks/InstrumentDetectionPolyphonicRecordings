#ifndef ONSET_DETECTOR_H
#define ONSET_DETECTOR_H

#include <stddef.h>

// Number of instruments in the probability timelines. Must match the model output.
#define NUM_INSTRUMENTS 11

// Onset detection settings for one instrument. Same parameters as the Python onset detector.
typedef struct {
    int lag_frames;
    float flux_threshold;
    int min_flux_period_frames;
    int min_activity_ms;
    int min_silence_ms_to_fill;
    float min_significance_score;
} OnsetParams;

// Finds the first onset of each instrument in the probability timelines and prints the results.
void find_and_print_onsets(const float* prob_timelines, int num_frames, const float* thresholds);
// Prints the periods of sustained activity for each instrument.
void find_and_print_activity_periods(const float* prob_timelines, int num_frames, const float* thresholds);

#endif // ONSET_DETECTOR_H