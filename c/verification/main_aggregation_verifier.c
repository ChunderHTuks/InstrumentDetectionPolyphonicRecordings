// Checks the C prediction aggregation and display logic against Python.
// Needs verification_data/py_full_raw_predictions.bin, written by the Python verification script.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"
#include "model_params.h"

#define VERIFICATION_DIR "./verification_data"
#define PYTHON_RAW_PREDS_FILE "./verification_data/py_full_raw_predictions.bin"
#define NUM_INSTRUMENTS 11
#define FRAME_WIDTH 44
#define SAMPLE_RATE 44100
#define CQT_HOP 512
#define INFERENCE_HOP_MS 500
#define FRAME_SIZE_MS 500
#define TOTAL_CQT_FRAMES 485 

void print_predictions(const float* preds_buffer, float thresholds[]);

// Loads the raw per-window predictions that Python produced (windows x instruments x 44 frames).
// Copies each window's predictions onto the CQT frames it covers.
// The windows overlap, so a CQT frame can collect several predictions.
// Averages the predictions of each CQT frame.
// Drops the frames that came from the reflection padding at both ends of the clip.
// Prints the instruments whose average is above their threshold for every remaining frame.
// The frame count and the thresholds are hard-coded for the one test clip used in the Python comparison.
int main() {
    printf("--- C Full Aggregation & Display Logic Verifier ---\n");

    printf("\n[Step 1] Loading Python's full raw predictions...\n");
    long num_elements;
    float* raw_preds_flat = load_binary_file(PYTHON_RAW_PREDS_FILE, &num_elements);
    if (!raw_preds_flat) return 1;

    long num_windows = num_elements / (NUM_INSTRUMENTS * FRAME_WIDTH);
    printf("  - Loaded data for %ld windows.\n", num_windows);

    // One list per CQT frame. predictions holds count vectors of NUM_INSTRUMENTS floats each.
    typedef struct {
        float* predictions;
        int count;
    } PredictionList;
    
    PredictionList* aggregated_preds = (PredictionList*)calloc(TOTAL_CQT_FRAMES, sizeof(PredictionList));
    if (!aggregated_preds) { return 1; }

    printf("\n[Step 2] Running C aggregation logic on Python data...\n");
    const float cqt_frame_duration_ms = (float)CQT_HOP / SAMPLE_RATE * 1000.0f;
    int frames_per_aggregation_hop = (int)roundf(INFERENCE_HOP_MS / cqt_frame_duration_ms);
    if (frames_per_aggregation_hop < 1) frames_per_aggregation_hop = 1;

    for (int i = 0; i < num_windows; ++i) {
        size_t start_frame = i * frames_per_aggregation_hop;
        
        for (int j = 0; j < FRAME_WIDTH; ++j) {
            size_t current_cqt_frame_idx = start_frame + j;

            if (current_cqt_frame_idx < TOTAL_CQT_FRAMES) {
                float prediction_vector[NUM_INSTRUMENTS];
                for (int inst = 0; inst < NUM_INSTRUMENTS; ++inst) {
                    long src_idx = (long)i * NUM_INSTRUMENTS * FRAME_WIDTH + (long)inst * FRAME_WIDTH + j;
                    prediction_vector[inst] = raw_preds_flat[src_idx];
                }

                PredictionList* list = &aggregated_preds[current_cqt_frame_idx];
                list->predictions = (float*)realloc(list->predictions, (list->count + 1) * NUM_INSTRUMENTS * sizeof(float));
                memcpy(list->predictions + list->count * NUM_INSTRUMENTS, prediction_vector, NUM_INSTRUMENTS * sizeof(float));
                list->count++;
            }
        }
    }
    
    printf("\n--- Final Averaged Predictions (C Logic on Python Data) ---\n");
    
    size_t pad_samples = (size_t)((FRAME_SIZE_MS / 2.0) / 1000.0 * SAMPLE_RATE);
    // Padding is half a window (250 ms) on each side, converted to CQT frames.
    size_t pad_frames = pad_samples / CQT_HOP; // Should be 21
    
    size_t start_frame_orig = pad_frames;
    size_t end_frame_orig = TOTAL_CQT_FRAMES - pad_frames;
    
    float thresholds[11] = {0.469, 0.335, 0.458, 0.511, 0.816, 0.772, 0.798, 0.883, 0.831, 0.831, 0.756};

    for (size_t i = start_frame_orig; i < end_frame_orig; ++i) {
        float avg_preds[11] = {0};
        PredictionList* list = &aggregated_preds[i];

        if (list->count > 0) {
            for (int p = 0; p < list->count; ++p) {
                for (int inst = 0; inst < NUM_INSTRUMENTS; ++inst) {
                    avg_preds[inst] += list->predictions[p * NUM_INSTRUMENTS + inst];
                }
            }
            for (int inst = 0; inst < NUM_INSTRUMENTS; ++inst) {
                avg_preds[inst] /= list->count;
            }
        }
        
        size_t display_frame_num = i - start_frame_orig;
        float time_sec = display_frame_num * cqt_frame_duration_ms / 1000.0f;
        
        printf("Frame %5zu [%8.3fs / %7ldms]: ", display_frame_num + 1, time_sec, (long)(time_sec * 1000));
        print_predictions(avg_preds, thresholds);
    }

    for(size_t i=0; i<TOTAL_CQT_FRAMES; ++i) {
        if (aggregated_preds[i].predictions) free(aggregated_preds[i].predictions);
    }
    free(aggregated_preds);
    free(raw_preds_flat);

    printf("\nSUCCESS: C aggregation verification complete.\n");
    return 0;
}
// Prints one line for a frame: every instrument whose averaged prediction is above its own threshold.
// Prints "No instruments" if none are above their threshold.
void print_predictions(const float* preds_buffer, float thresholds[]) {
    const char* instrument_labels[] = {"Piano", "Violin", "Cello", "Viola", "Contrabass", 
                                       "Clarinet", "Bassoon", "Oboe", "Flute", 
                                       "Fr Horn", "Trumpet"};
    int detected_count = 0;
    
    for (int i = 0; i < 11; ++i) {
        if (preds_buffer[i] > thresholds[i]) {
            if (detected_count > 0) printf(", ");
            printf("%s (AVG: %.3f)", instrument_labels[i], preds_buffer[i]);
            detected_count++;
        }
    }

    if (detected_count == 0) {
        printf("No instruments");
    }
    printf("\n");
}