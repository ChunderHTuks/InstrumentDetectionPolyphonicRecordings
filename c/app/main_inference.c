#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <sys/time.h>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "utils.h"
#include "cpu_layers.h"
#include "weight_loader.h"
#include "model_params.h"
#include "cqt.h"
#include "onset_detector.h"
#include "sound_acquisition.h"

// Capture device used in record mode. The card and device numbers depend on the Jetson setup.
#define ALSA_DEVICE "plughw:2,0"
#define INPUT_WAV_FILE "2083_short.wav"
#define SAMPLE_RATE 44100
#define NUM_INSTRUMENTS 11
// The HCQT stacks one CQT per harmonic, where harmonic h starts at FMIN * (h + 1).
// 84 bins at 12 per octave covers 7 octaves from FMIN (about C1).
#define N_HARMONICS 5
#define FMIN 32.7f
#define N_BINS 84
#define BINS_PER_OCTAVE 12
#define HOP_LENGTH 512
// One model input covers 44 CQT frames, which is about 500 ms (44 * 512 / 44100).
#define FRAME_WIDTH 44
#define INFERENCE_HOP_MS 500
#define FRAME_SIZE_MS 500

// Mono audio samples.
typedef struct {
    float* data;
    size_t num_samples;
} AudioBuffer;

// HCQT magnitudes laid out as [harmonic][bin][frame].
typedef struct {
    float* data;
    size_t num_frames;
} HCQTBuffer;

// Everything loaded from the weight files.
// head_weights[head][0] is the first conv of the head and [head][1] is the classifier.
// The heads are in the order piano, strings, woodwinds, brass.
typedef struct {
    InstanceNormWeights norm_weights;
    ConvWithBias backbone_weights[NUM_BACKBONE_LAYERS];
    ConvWithBias head_weights[4][2];
} ModelWeights;

// All predictions made for one CQT frame, 11 values each.
// Windows overlap, so a frame receives one prediction from each window that covers it.
typedef struct {
    float* predictions;
    int count;
    int capacity;
} PredictionList;

// One PredictionList for every CQT frame of the padded clip.
typedef struct {
    PredictionList* lists;
    size_t num_frames;
} AggregationBuffer;

// Reads a WAV file as float samples.
// The sample rate must be 44100 Hz because the CQT settings assume it.
// Returns 1 on success and 0 on failure.
int load_audio(const char* filename, AudioBuffer* audio) {
    printf("\n[1] Loading WAV file: %s\n", filename);
    unsigned int channels, sampleRate;
    drwav_uint64 totalFrameCount;
    audio->data = drwav_open_file_and_read_pcm_frames_f32(filename, &channels, &sampleRate, &totalFrameCount, NULL);
    if (!audio->data) { fprintf(stderr, " -> ERROR: Could not open WAV file.\n"); return 0; }
    if (sampleRate != SAMPLE_RATE) { fprintf(stderr, " -> ERROR: Sample rate mismatch.\n"); return 0; }
    if (channels != 1) { fprintf(stderr, " -> WARNING: Audio is not mono. Only first channel will be used implicitly.\n"); }
    audio->num_samples = (size_t)totalFrameCount;
    printf(" -> Loaded %.2f seconds of audio.\n", (float)audio->num_samples / SAMPLE_RATE);
    return 1;
}

// Pads 250 ms of mirrored audio on both ends.
// This lets the first and last real frames be covered by full 500 ms windows.
// The mirror skips the edge sample itself, the same as reflection padding in PyTorch.
int apply_padding(const AudioBuffer* audio_in, AudioBuffer* audio_out) {
    printf("\n[2] Applying reflection padding...\n");
    size_t pad_samples = (size_t)((FRAME_SIZE_MS / 2.0) / 1000.0 * SAMPLE_RATE);
    audio_out->num_samples = audio_in->num_samples + 2 * pad_samples;
    audio_out->data = (float*)malloc(audio_out->num_samples * sizeof(float));
    if (!audio_out->data) { fprintf(stderr, " -> ERROR: Failed to allocate padded buffer.\n"); return 0; }

    for (size_t i = 0; i < pad_samples; ++i) {
        audio_out->data[pad_samples - 1 - i] = audio_in->data[i + 1];
        audio_out->data[pad_samples + audio_in->num_samples + i] = audio_in->data[audio_in->num_samples - 2 - i];
    }
    memcpy(audio_out->data + pad_samples, audio_in->data, audio_in->num_samples * sizeof(float));
    printf(" -> Padded audio to %zu samples.\n", audio_out->num_samples);
    return 1;
}

// Computes the HCQT of the whole clip once, so the windows can be cut from it afterwards.
// Each harmonic is a CQT with fmin multiplied by the harmonic number, and only the magnitude is kept.
// The buffer is allocated after the first CQT, since only then is the frame count known.
int compute_full_hcqt(const AudioBuffer* audio, HCQTBuffer* hcqt) {
    printf("\n[3] Computing full HCQT.\n");
    hcqt->num_frames = 0;
    hcqt->data = NULL;
    for (int h = 0; h < N_HARMONICS; ++h) {
        printf("  - Harmonic %d/%d\n", h + 1, N_HARMONICS);
        CQTResult* cqt_res = compute_cqt(audio->data, audio->num_samples, SAMPLE_RATE, HOP_LENGTH, FMIN * (h + 1), N_BINS, BINS_PER_OCTAVE);
        if (!cqt_res) { fprintf(stderr, " -> ERROR: CQT computation failed.\n"); return 0; }
        if (h == 0) {
            hcqt->num_frames = cqt_res->time_frames;
            hcqt->data = (float*)malloc(N_HARMONICS * N_BINS * hcqt->num_frames * sizeof(float));
            if (!hcqt->data) { fprintf(stderr, " -> ERROR: Failed to allocate HCQT buffer.\n"); return 0; }
        }
        for (size_t i = 0; i < cqt_res->freq_bins * hcqt->num_frames; ++i) {
            hcqt->data[h * N_BINS * hcqt->num_frames + i] = cabsf(cqt_res->data[i]);
        }
        free_cqt_result(cqt_res);
    }
    printf(" -> HCQT computed with shape [H=%d, B=%d, T=%zu].\n", N_HARMONICS, N_BINS, hcqt->num_frames);
    return 1;
}

// Loads the instance norm, the six backbone layers and the two layers of each of the four heads.
// The numbers in pytorch_indices are the layer positions in the PyTorch Sequential.
int load_all_weights(ModelWeights* weights) {
    printf("\n[5] Loading all model weights...\n");
    weights->norm_weights = load_instance_norm_weights();
    int pytorch_indices[] = {0, 3, 6, 9, 12, 15};
    for (int i = 0; i < NUM_BACKBONE_LAYERS; ++i) {
        weights->backbone_weights[i] = load_folded_conv_block("backbone", pytorch_indices[i]);
    }
    weights->head_weights[0][0] = load_folded_conv_block("piano", 0); weights->head_weights[0][1] = load_conv_with_bias("piano", "classifier");
    weights->head_weights[1][0] = load_folded_conv_block("strings", 0); weights->head_weights[1][1] = load_conv_with_bias("strings", "classifier");
    weights->head_weights[2][0] = load_folded_conv_block("woodwinds", 0); weights->head_weights[2][1] = load_conv_with_bias("woodwinds", "classifier");
    weights->head_weights[3][0] = load_folded_conv_block("brass", 0); weights->head_weights[3][1] = load_conv_with_bias("brass", "classifier");
    printf(" -> All weights loaded successfully.\n");
    return 1;
}

// Runs the full model on one 500 ms patch.
// Steps:
// 1. Copy the patch into buffer A and apply instance normalisation.
// 2. Run the backbone layers, switching between buffer A and buffer B so no layer needs a new allocation.
// 3. Feed the backbone output to each of the four heads, which each have a conv, a classifier conv and a sigmoid.
// The head outputs hold probabilities for that head's instruments across the 44 frames.
// The buffers are sized for the largest layer output.
void run_inference_on_frame(float* frame_data, const ModelWeights* weights, float* head_outputs[]) {
    // Room for the largest layer output, 32 channels by 44 frames by 84 bins.
    long max_buffer_elements = 32 * 44 * 84;
    float* bufferA = (float*)aligned_malloc(max_buffer_elements * sizeof(float), SIMD_ALIGNMENT);
    float* bufferB = (float*)aligned_malloc(max_buffer_elements * sizeof(float), SIMD_ALIGNMENT);
    memcpy(bufferA, frame_data, N_HARMONICS * FRAME_WIDTH * N_BINS * sizeof(float));

    instance_norm_cpu_opt(bufferA, N_HARMONICS, FRAME_WIDTH * N_BINS, &weights->norm_weights);

    const float* current_input = bufferA;
    float* current_output = bufferB;
    LayerOptParams backbone_opts[NUM_BACKBONE_LAYERS];
    for(int i=0; i<NUM_BACKBONE_LAYERS; ++i) precompute_layer_opt_params(&backbone_opts[i], &backbone_architecture[i]);
    for (int i = 0; i < NUM_BACKBONE_LAYERS; ++i) {
        // Even layers read from A and write to B, odd layers do the opposite.
        if (i % 2 == 0) { current_input = bufferA; current_output = bufferB; }
        else { current_input = bufferB; current_output = bufferA; }
        conv_relu_block(current_input, current_output, &backbone_architecture[i], &backbone_opts[i], &weights->backbone_weights[i]);
    }
    const float* backbone_out = current_output;
    
    LayerOptParams head_opts[4][2];
    precompute_layer_opt_params(&head_opts[0][0], &piano_head_architecture[0]); precompute_layer_opt_params(&head_opts[0][1], &piano_head_architecture[1]);
    precompute_layer_opt_params(&head_opts[1][0], &strings_head_architecture[0]); precompute_layer_opt_params(&head_opts[1][1], &strings_head_architecture[1]);
    precompute_layer_opt_params(&head_opts[2][0], &woodwinds_head_architecture[0]); precompute_layer_opt_params(&head_opts[2][1], &woodwinds_head_architecture[1]);
    precompute_layer_opt_params(&head_opts[3][0], &brass_head_architecture[0]); precompute_layer_opt_params(&head_opts[3][1], &brass_head_architecture[1]);

    const LayerParams* head_archs[][2] = {{&piano_head_architecture[0],&piano_head_architecture[1]},{&strings_head_architecture[0],&strings_head_architecture[1]},{&woodwinds_head_architecture[0],&woodwinds_head_architecture[1]},{&brass_head_architecture[0],&brass_head_architecture[1]}};
    long agg_out_size = head_opts[0][0].total_output_size;
    float* agg_out = (float*)aligned_malloc(agg_out_size * sizeof(float), SIMD_ALIGNMENT);
    for(int i=0; i<4; ++i) {
        conv_relu_block(backbone_out, agg_out, head_archs[i][0], &head_opts[i][0], &weights->head_weights[i][0]);
        conv2d_cpu_opt(agg_out, head_outputs[i], head_archs[i][1], weights->head_weights[i][1].weight, weights->head_weights[i][1].bias, &head_opts[i][1]);
        sigmoid_cpu_opt(head_outputs[i], head_opts[i][1].total_output_size);
    }
    
    aligned_free(bufferA);
    aligned_free(bufferB);
    aligned_free(agg_out);
}

// Stores the 44 frame predictions of one window in the lists of the CQT frames they belong to.
// The 11 values per frame follow the label order: piano, then 4 strings, then 4 woodwinds, then 2 brass.
void aggregate_predictions(int window_idx, float* head_outputs[], AggregationBuffer* agg_buffer) {
    const float cqt_frame_duration_ms = (float)HOP_LENGTH / SAMPLE_RATE * 1000.0f;
    int frames_per_aggregation_hop = (int)roundf(INFERENCE_HOP_MS / cqt_frame_duration_ms);
    if (frames_per_aggregation_hop < 1) frames_per_aggregation_hop = 1;
    
    size_t start_frame = window_idx * frames_per_aggregation_hop;

    for (int j = 0; j < FRAME_WIDTH; ++j) {
        size_t current_cqt_frame_idx = start_frame + j;
        if (current_cqt_frame_idx < agg_buffer->num_frames) {
            PredictionList* list = &agg_buffer->lists[current_cqt_frame_idx];
            if (list->count >= list->capacity) {
                list->capacity = (list->capacity == 0) ? 2 : list->capacity * 2;
                list->predictions = (float*)realloc(list->predictions, list->capacity * NUM_INSTRUMENTS * sizeof(float));
            }
            float* dest = list->predictions + list->count * NUM_INSTRUMENTS;
            int inst_idx = 0;
            dest[inst_idx++] = head_outputs[0][j];
            for (int k = 0; k < 4; ++k) dest[inst_idx++] = head_outputs[1][k * FRAME_WIDTH + j];
            for (int k = 0; k < 4; ++k) dest[inst_idx++] = head_outputs[2][k * FRAME_WIDTH + j];
            for (int k = 0; k < 2; ++k) dest[inst_idx++] = head_outputs[3][k * FRAME_WIDTH + j];
            list->count++;
        }
    }
}

// Averages the predictions of every frame and prints the instruments above their thresholds.
// The padded frames at the start and end are skipped so the times match the original clip.
void print_final_results(const AggregationBuffer* agg_buffer) {
    printf("\nFinal Averaged Predictions\n");
    const float cqt_frame_duration_ms = (float)HOP_LENGTH / SAMPLE_RATE * 1000.0f;
    size_t pad_samples = (size_t)((FRAME_SIZE_MS / 2.0) / 1000.0 * SAMPLE_RATE);
    size_t pad_frames = pad_samples / HOP_LENGTH;
    size_t start_frame_orig = pad_frames;
    size_t end_frame_orig = agg_buffer->num_frames - pad_frames;
    // One threshold per instrument, in label order.
    float thresholds[11] = {0.116, 0.295, 0.296, 0.423, 0.895, 0.745, 0.89,  0.886, 0.859, 0.828, 0.536};
    
    for (size_t i = start_frame_orig; i < end_frame_orig; ++i) {
        float avg_preds[11] = {0};
        PredictionList* list = &agg_buffer->lists[i];
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
}

// Prints the names and probabilities of the instruments whose probability is above their threshold.
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
        printf("No instruments.");
    }
    printf("\n");
}

// Prints a duration in milliseconds as minutes, seconds and milliseconds.
void print_formatted_time(const char* label, double total_ms) {
    long minutes = (long)(total_ms / 1000.0) / 60;
    long seconds = (long)(total_ms / 1000.0) % 60;
    long milliseconds = (long)total_ms % 1000;
    
    printf("%-35s %02ldm %02lds %03ldms\n", label, minutes, seconds, milliseconds);
}

// Runs the whole pipeline.
// Steps:
// 1. Get audio, either from the microphone with the argument "record" or from the WAV file.
// 2. Pad the audio and compute the HCQT.
// 3. Load the weights.
// 4. Slide a 500 ms window over the HCQT with a hop of 500 ms and run the model on each window.
// 5. Average the overlapping predictions for every frame and print them.
// 6. Run the onset detection and activity period detection on the averaged probabilities.
// 7. Print the timing and free everything.
int main(int argc, char *argv[]) {
    struct timeval start_time, end_time, total_start_time, total_end_time;
    double hcqt_time_ms = 0;
    double model_time_ms = 0;

    int record_mode = 0;
    if (argc > 1 && strcmp(argv[1], "record") == 0) {
        record_mode = 1;
    }
    
    AudioBuffer audio_orig;

    if (record_mode) {
        printf("\n[1] Starting audio acquisition from device: %s\n", ALSA_DEVICE);
        audio_orig.data = record_audio_to_float(ALSA_DEVICE, SAMPLE_RATE, &audio_orig.num_samples);
        if (!audio_orig.data || audio_orig.num_samples == 0) {
            fprintf(stderr, " -> ERROR: Failed to record audio.\n");
            return 1;
        }
    } else {
        if (!load_audio(INPUT_WAV_FILE, &audio_orig)) return 1;
    }

    // The total time starts after the audio is available, so the recording time is not counted.
    gettimeofday(&total_start_time, NULL);

    AudioBuffer audio_padded;
    if (!apply_padding(&audio_orig, &audio_padded)) return 1;
    free(audio_orig.data);
    
    gettimeofday(&start_time, NULL);
    HCQTBuffer hcqt;
    if (!compute_full_hcqt(&audio_padded, &hcqt)) return 1;
    gettimeofday(&end_time, NULL);
    hcqt_time_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0;
    hcqt_time_ms += (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    free(audio_padded.data);

    ModelWeights weights;
    if (!load_all_weights(&weights)) return 1;

    // Output buffers for the four heads, sized from the classifier layer of each head.
    float* head_outputs[4];
    head_outputs[0] = (float*)aligned_malloc(piano_head_architecture[1].c_out * piano_head_architecture[1].h_out * sizeof(float), SIMD_ALIGNMENT);
    head_outputs[1] = (float*)aligned_malloc(strings_head_architecture[1].c_out * strings_head_architecture[1].h_out * sizeof(float), SIMD_ALIGNMENT);
    head_outputs[2] = (float*)aligned_malloc(woodwinds_head_architecture[1].c_out * woodwinds_head_architecture[1].h_out * sizeof(float), SIMD_ALIGNMENT);
    head_outputs[3] = (float*)aligned_malloc(brass_head_architecture[1].c_out * brass_head_architecture[1].h_out * sizeof(float), SIMD_ALIGNMENT);
    float* frame_buffer = (float*)aligned_malloc(N_HARMONICS * FRAME_WIDTH * N_BINS * sizeof(float), SIMD_ALIGNMENT);

    AggregationBuffer agg_buffer;
    agg_buffer.num_frames = hcqt.num_frames;
    agg_buffer.lists = (PredictionList*)calloc(agg_buffer.num_frames, sizeof(PredictionList));
    
    const float cqt_frame_duration_ms = (float)HOP_LENGTH / SAMPLE_RATE * 1000.0f;
    int frames_per_extraction_hop = (int)ceilf(INFERENCE_HOP_MS / cqt_frame_duration_ms);
    if (frames_per_extraction_hop < 1) frames_per_extraction_hop = 1;
    long num_inference_windows = (hcqt.num_frames >= FRAME_WIDTH) ? ((hcqt.num_frames - FRAME_WIDTH) / frames_per_extraction_hop) + 1 : 0;
    
    printf("\nProcessing %ld patches of size 500ms.\n", num_inference_windows);
    for (int i = 0; i < num_inference_windows; ++i) {
        // Copies the window from [harmonic][bin][frame] into the [harmonic][frame][bin] order the model expects.
        size_t frame_start_index = i * frames_per_extraction_hop;

        for (int h=0; h<N_HARMONICS; ++h) for (int t=0; t<FRAME_WIDTH; ++t) for (int b=0; b<N_BINS; ++b) {
            long src_idx = (long)h*N_BINS*hcqt.num_frames + (long)b*hcqt.num_frames + (frame_start_index + t);
            long dest_idx = (long)h*FRAME_WIDTH*N_BINS + (long)t*N_BINS + b;
            frame_buffer[dest_idx] = hcqt.data[src_idx];
        }

        gettimeofday(&start_time, NULL);
        run_inference_on_frame(frame_buffer, &weights, head_outputs);
        gettimeofday(&end_time, NULL);
        model_time_ms += (end_time.tv_sec - start_time.tv_sec) * 1000.0;
        model_time_ms += (end_time.tv_usec - start_time.tv_usec) / 1000.0;

        aggregate_predictions(i, head_outputs, &agg_buffer);
    }

    print_final_results(&agg_buffer);

    // Builds one averaged probability timeline for the original clip, without the padded frames, as input to the onset logic.
    size_t pad_samples = (size_t)((FRAME_SIZE_MS / 2.0) / 1000.0 * SAMPLE_RATE);
    size_t pad_frames = pad_samples / HOP_LENGTH;
    size_t start_frame_orig = pad_frames;
    size_t end_frame_orig = agg_buffer.num_frames - pad_frames;
    int num_orig_frames = end_frame_orig - start_frame_orig;

    float* final_prob_timelines = (float*)malloc(num_orig_frames * NUM_INSTRUMENTS * sizeof(float));
    
    for (int i = 0; i < num_orig_frames; ++i) {
        float avg_preds[11] = {0};
        PredictionList* list = &agg_buffer.lists[start_frame_orig + i];
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
        memcpy(final_prob_timelines + i * NUM_INSTRUMENTS, avg_preds, NUM_INSTRUMENTS * sizeof(float));
    }

    // Same thresholds as in print_final_results.
    float thresholds[11] = {0.116, 0.295, 0.296, 0.423, 0.895, 0.745, 0.89,  0.886, 0.859, 0.828, 0.536};
    find_and_print_onsets(final_prob_timelines, num_orig_frames, thresholds);
    find_and_print_activity_periods(final_prob_timelines, num_orig_frames, thresholds);
    free(final_prob_timelines);

    gettimeofday(&total_end_time, NULL);
    double total_time_ms = (total_end_time.tv_sec - total_start_time.tv_sec) * 1000.0;
    total_time_ms += (total_end_time.tv_usec - total_start_time.tv_usec) / 1000.0;

    printf("\n" "Inference time statistics\n");
    print_formatted_time("Time for HCQT computation:", hcqt_time_ms);
    print_formatted_time("Time for all model forward passes:", model_time_ms);
    if (num_inference_windows > 0) {
        printf("%-35s %.2f ms\n", "Average time per frame:", model_time_ms / num_inference_windows);
    }
    print_formatted_time("Total application time:", total_time_ms);
    printf("---------------------------\n");

    printf("\n[Cleanup] Releasing all resources...\n");
    free(hcqt.data);
    aligned_free(frame_buffer);
    for(int i=0; i<4; ++i) aligned_free(head_outputs[i]);
    for(size_t i=0; i<agg_buffer.num_frames; ++i) if(agg_buffer.lists[i].predictions) free(agg_buffer.lists[i].predictions);
    free(agg_buffer.lists);
    
    free_instance_norm_weights(&weights.norm_weights);
    for (int i = 0; i < NUM_BACKBONE_LAYERS; ++i) free_conv_with_bias(&weights.backbone_weights[i]);
    for(int i=0; i<4; ++i) for(int j=0; j<2; ++j) free_conv_with_bias(&weights.head_weights[i][j]);

    cleanup_cqt_caches();
    printf(" -> Cleanup complete.\n");

    return 0;
}