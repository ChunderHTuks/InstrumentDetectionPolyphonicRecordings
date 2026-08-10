#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "onset_detector.h"

#define CQT_HOP 512
#define SAMPLE_RATE 44100

typedef struct {
    float start_time;
    float end_time;
    float value;
} TimePeriod;

typedef struct {
    TimePeriod* periods;
    int count;
    int capacity;
} PeriodList;

static void add_period(PeriodList* list, float start, float end, float value) {
    if (list->count >= list->capacity) {
        list->capacity = (list->capacity == 0) ? 8 : list->capacity * 2;
        list->periods = (TimePeriod*)realloc(list->periods, list->capacity * sizeof(TimePeriod));
    }
    list->periods[list->count].start_time = start;
    list->periods[list->count].end_time = end;
    list->periods[list->count].value = value;
    list->count++;
}

static void free_period_list(PeriodList* list) {
    if (list && list->periods) {
        free(list->periods);
        list->periods = NULL;
        list->count = 0;
        list->capacity = 0;
    }
}

static void filter_state_timeline(int* timeline, int num_frames, int min_duration_frames, int target_state, int new_state) {
    if (num_frames == 0 || min_duration_frames <= 0) return;
    int* temp_timeline = (int*)malloc(num_frames * sizeof(int));
    memcpy(temp_timeline, timeline, num_frames * sizeof(int));
    int is_in_run = 0;
    int run_start_index = 0;
    for (int i = 0; i < num_frames; ++i) {
        if (temp_timeline[i] == target_state && !is_in_run) {
            is_in_run = 1;
            run_start_index = i;
        } else if (temp_timeline[i] != target_state && is_in_run) {
            is_in_run = 0;
            int duration = i - run_start_index;
            if (duration < min_duration_frames && run_start_index > 0) {
                for (int j = run_start_index; j < i; ++j) timeline[j] = new_state;
            }
        }
    }
    if (is_in_run) {
        int duration = num_frames - run_start_index;
        if (duration < min_duration_frames && run_start_index > 0) {
            for (int j = run_start_index; j < num_frames; ++j) timeline[j] = new_state;
        }
    }
    free(temp_timeline);
}

static PeriodList find_activity_periods(const float* prob_timeline, int num_frames, float threshold, int min_silence_ms, int min_activity_ms) {
    PeriodList list = {NULL, 0, 0};
    if (num_frames == 0) return list;

    const float frame_duration_s = (float)CQT_HOP / SAMPLE_RATE;
    int min_activity_frames = (int)(min_activity_ms / (1000.0f * frame_duration_s));
    int min_silence_frames = (int)(min_silence_ms / (1000.0f * frame_duration_s));
    
    int* s_timeline = (int*)malloc(num_frames * sizeof(int));
    for (int i = 0; i < num_frames; ++i) {
        s_timeline[i] = (prob_timeline[i] >= threshold) ? 1 : 0;
    }
    
    filter_state_timeline(s_timeline, num_frames, min_silence_frames, 0, 1);
    filter_state_timeline(s_timeline, num_frames, min_activity_frames, 1, 0);

    if (num_frames > 0 && s_timeline[0] == 1) {
        int k = 0;
        while(k < num_frames && s_timeline[k] == 1) s_timeline[k++] = 0;
    }

    int* start_indices = (int*)malloc(num_frames * sizeof(int));
    int* end_indices = (int*)malloc(num_frames * sizeof(int));
    int start_count = 0;
    int end_count = 0;

    if (num_frames > 0 && s_timeline[0] == 1) start_indices[start_count++] = 0;
    for (int j = 1; j < num_frames; ++j) {
        if (s_timeline[j] == 1 && s_timeline[j-1] == 0) start_indices[start_count++] = j;
        else if (s_timeline[j] == 0 && s_timeline[j-1] == 1) end_indices[end_count++] = j;
    }
    if (num_frames > 0 && s_timeline[num_frames - 1] == 1) end_indices[end_count++] = num_frames;
    
    for (int k = 0; k < start_count; ++k) {
        int start_idx = start_indices[k];
        int end_idx = end_indices[k];
        float sum_prob = 0.0f;
        for(int p=start_idx; p < end_idx; ++p) sum_prob += prob_timeline[p];
        float avg_prob = (end_idx > start_idx) ? sum_prob / (end_idx - start_idx) : 0.0f;
        add_period(&list, start_idx * frame_duration_s, (end_idx - 1) * frame_duration_s, avg_prob);
    }
    
    free(s_timeline);
    free(start_indices);
    free(end_indices);
    return list;
}

static PeriodList find_flux_periods(const float* prob_timeline, int num_frames, const OnsetParams* params) {
    PeriodList list = {NULL, 0, 0};
    if (num_frames == 0) return list;
    const float frame_duration_s = (float)CQT_HOP / SAMPLE_RATE;
    float* rectified_flux = (float*)malloc(num_frames * sizeof(float));
    int* high_flux_state = (int*)malloc(num_frames * sizeof(int));

    for (int i = 0; i < num_frames; ++i) {
        float lagged_prob = (i >= params->lag_frames) ? prob_timeline[i - params->lag_frames] : 0.0f;
        rectified_flux[i] = fmaxf(0.0f, prob_timeline[i] - lagged_prob);
        high_flux_state[i] = (rectified_flux[i] > params->flux_threshold) ? 1 : 0;
    }
    filter_state_timeline(high_flux_state, num_frames, params->min_flux_period_frames, 1, 0);

    if (num_frames > 0 && high_flux_state[0] == 1) high_flux_state[0] = 0;
    for (int i = 1; i < num_frames; ++i) {
        if (high_flux_state[i] == 1 && high_flux_state[i-1] == 0) {
            int start_idx = i;
            int end_idx = i;
            while (end_idx < num_frames && high_flux_state[end_idx] == 1) end_idx++;
            float max_flux = -1.0f;
            int peak_local_idx = 0;
            for(int k = start_idx; k < end_idx; ++k) {
                if (rectified_flux[k] > max_flux) {
                    max_flux = rectified_flux[k];
                    peak_local_idx = k - start_idx;
                }
            }
            int peak_global_idx = start_idx + peak_local_idx;
            add_period(&list, start_idx * frame_duration_s, (end_idx - 1) * frame_duration_s, peak_global_idx * frame_duration_s);
            i = end_idx - 1;
        }
    }
    free(rectified_flux);
    free(high_flux_state);
    return list;
}

void find_and_print_activity_periods(const float* prob_timelines, int num_frames, const float* thresholds) {
    const char* instrument_labels[] = {"Piano", "Violin", "Cello", "Viola", "Contrabass", "Clarinet", "Bassoon", "Oboe", "Flute", "Fr Horn", "Trumpet"};
    const int min_activity_ms = 1000;
    const int min_silence_ms_to_fill = 500;
    const float min_significance_score = 0.04;
    const float frame_duration_s = (float)CQT_HOP / SAMPLE_RATE;
    const float clip_duration = (num_frames > 0) ? num_frames * frame_duration_s : 0.0f;

    PeriodList final_periods[NUM_INSTRUMENTS] = {0};
    float* instrument_timeline = (float*)malloc(num_frames * sizeof(float));

    for (int i = 0; i < NUM_INSTRUMENTS; ++i) {
        for (int j = 0; j < num_frames; ++j) instrument_timeline[j] = prob_timelines[j * NUM_INSTRUMENTS + i];
        
        PeriodList candidate_periods = find_activity_periods(instrument_timeline, num_frames, thresholds[i], min_silence_ms_to_fill, min_activity_ms);

        for (int k = 0; k < candidate_periods.count; ++k) {
            TimePeriod* p = &candidate_periods.periods[k];
            float period_duration = p->end_time - p->start_time;
            if(period_duration < 0) period_duration = 0;
            float duration_factor = (clip_duration > 0) ? period_duration / clip_duration : 0.0f;
            float significance_score = p->value * duration_factor;

            if (significance_score >= min_significance_score) {
                add_period(&final_periods[i], p->start_time, p->end_time, p->value);
            }
        }
        free_period_list(&candidate_periods);
    }
    free(instrument_timeline);

    printf("\n\n" "================================================================================" "\n");
    printf("Instrument activity periods of significance\n");
    printf("================================================================================" "\n");
    for (int i = 0; i < NUM_INSTRUMENTS; ++i) {
        if (final_periods[i].count > 0) {
            printf("%s activity periods:\n", instrument_labels[i]);
            for (int j = 0; j < final_periods[i].count; ++j) {
                TimePeriod* p = &final_periods[i].periods[j];
                printf("  %7.3fs - %7.3fs (Average: %.2f)\n", p->start_time, p->end_time, p->value);
            }
        }
    }
    printf("================================================================================" "\n");

    for (int i = 0; i < NUM_INSTRUMENTS; ++i) free_period_list(&final_periods[i]);
}

void find_and_print_onsets(const float* prob_timelines, int num_frames, const float* thresholds) {
    const char* instrument_labels[] = {"Piano", "Violin", "Cello", "Viola", "Contrabass", "Clarinet", "Bassoon", "Oboe", "Flute", "Fr Horn", "Trumpet"};
    OnsetParams params[NUM_INSTRUMENTS] = {
        {2, 0.11, 2, 250, 600, 0.05}, {3, 0.10, 3, 250, 600, 0.01}, {2, 0.07, 2, 250, 500, 0.05},
        {2, 0.07, 2, 250, 500, 0.05}, {2, 0.11, 2, 250, 600, 0.05}, {3, 0.07, 5, 250, 400, 0.05},
        {6, 0.09, 5, 250, 500, 0.11}, {2, 0.07, 4, 250, 400, 0.05}, {2, 0.07, 4, 250, 500, 0.05},
        {2, 0.07, 2, 250, 500, 0.08}, {2, 0.11, 2, 250, 600, 0.05}
    };
    PeriodList activity_periods[NUM_INSTRUMENTS] = {0};
    PeriodList flux_periods[NUM_INSTRUMENTS] = {0};
    float* instrument_timeline = (float*)malloc(num_frames * sizeof(float));

    for (int i = 0; i < NUM_INSTRUMENTS; ++i) {
        for (int j = 0; j < num_frames; ++j) instrument_timeline[j] = prob_timelines[j * NUM_INSTRUMENTS + i];
        activity_periods[i] = find_activity_periods(instrument_timeline, num_frames, thresholds[i], params[i].min_silence_ms_to_fill, params[i].min_activity_ms);
        flux_periods[i] = find_flux_periods(instrument_timeline, num_frames, &params[i]);
    }
    free(instrument_timeline);

    float clip_duration = (num_frames > 0) ? num_frames * ((float)CQT_HOP / SAMPLE_RATE) : 0.0f;
    float final_onsets[NUM_INSTRUMENTS];
    for (int i=0; i<NUM_INSTRUMENTS; ++i) final_onsets[i] = -1.0f;

    for (int i = 0; i < NUM_INSTRUMENTS; ++i) {
        PeriodList* activity_list = &activity_periods[i];
        PeriodList* flux_list = &flux_periods[i];
        if (activity_list->count == 0 || flux_list->count == 0) continue;
        for (int j = 0; j < activity_list->count; ++j) {
            TimePeriod* activity = &activity_list->periods[j];
            float period_duration = activity->end_time - activity->start_time;
            if (period_duration < 0) period_duration = 0;
            float duration_factor = (clip_duration > 0) ? period_duration / clip_duration : 0;
            float significance_score = activity->value * duration_factor;
            if (significance_score >= params[i].min_significance_score) {
                for (int k = 0; k < flux_list->count; ++k) {
                    TimePeriod* flux = &flux_list->periods[k];
                    if (fmaxf(flux->start_time, activity->start_time) <= fminf(flux->end_time, activity->end_time)) {
                        float candidate_onset = flux->value;
                        if (final_onsets[i] < 0 || candidate_onset < final_onsets[i]) {
                            final_onsets[i] = candidate_onset;
                        }
                        break; 
                    }
                }
            }
        }
    }
    
    printf("\n" "Detected onsets within the mix are as follows:" "\n");
    for (int i = 0; i < NUM_INSTRUMENTS; i++) {
        if (final_onsets[i] >= 0) printf("  %-15s: %.3fs\n", instrument_labels[i], final_onsets[i]);
    }

    for (int i = 0; i < NUM_INSTRUMENTS; ++i) {
        free_period_list(&activity_periods[i]);
        free_period_list(&flux_periods[i]);
    }
}