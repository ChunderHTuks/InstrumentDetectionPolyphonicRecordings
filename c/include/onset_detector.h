#ifndef ONSET_DETECTOR_H
#define ONSET_DETECTOR_H

#include <stddef.h>

#define NUM_INSTRUMENTS 11

typedef struct {
    int lag_frames;
    float flux_threshold;
    int min_flux_period_frames;
    int min_activity_ms;
    int min_silence_ms_to_fill;
    float min_significance_score;
} OnsetParams;

void find_and_print_onsets(const float* prob_timelines, int num_frames, const float* thresholds);
void find_and_print_activity_periods(const float* prob_timelines, int num_frames, const float* thresholds);

#endif // ONSET_DETECTOR_H