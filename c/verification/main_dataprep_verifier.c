// Generates the C side of the data preparation check: padded audio, full HCQT and the per-window HCQT frames.
// Needs INPUT_WAV_FILE in the folder it is run from and an existing verification_data/ folder.
// The files it writes are compared against the Python versions by the Python verification script.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "utils.h"
#include "cpu_layers.h"
#include "model_params.h"
#include "cqt.h"

#define INPUT_WAV_FILE "1729.wav"
#define VERIFICATION_DIR "./verification_data"
#define SAMPLE_RATE 44100
#define N_HARMONICS 5
#define FMIN 32.7f
#define N_BINS 84
#define BINS_PER_OCTAVE 12
#define HOP_LENGTH 512
#define FRAME_WIDTH 44
#define INFERENCE_HOP_MS 500
#define FRAME_PAD_MS 250

// Reads the wav file and saves the raw samples to c_01_raw_audio.bin.
// Pads both ends with a reflection of the signal and saves it to c_02_padded_audio.bin.
// The reflection leaves out the edge sample itself, so the copy starts at index 1 and ends at the second last sample.
// Computes one CQT per harmonic with the minimum frequency set to FMIN times the harmonic number.
// Stores the magnitudes as [harmonic][bin][time] and saves them to c_03_full_hcqt.bin.
// Cuts the HCQT into windows of FRAME_WIDTH CQT frames, moving forward by about INFERENCE_HOP_MS each time.
// Reorders each window to [harmonic][time][bin], which is the layout the model input uses, and saves it to c_frame_NNN.bin.
// Windows that would run past the end of the HCQT are not saved.
int main() {
    printf("--- C Data Preparation Verification Generator ---\n");
    
    printf("\n[Step 1] Loading WAV file: %s\n", INPUT_WAV_FILE);
    unsigned int channels, sampleRate;
    drwav_uint64 totalFrameCount;
    float* audio_data_orig = drwav_open_file_and_read_pcm_frames_f32(INPUT_WAV_FILE, &channels, &sampleRate, &totalFrameCount, NULL);
    if (!audio_data_orig) { fprintf(stderr, "Error opening WAV file.\n"); return 1; }
    if (sampleRate != SAMPLE_RATE) { fprintf(stderr, "Error: Incorrect sample rate.\n"); return 1; }
    size_t num_samples_orig = (size_t)totalFrameCount;
    
    printf("  -> Saving raw audio float data...\n");
    write_binary_file(VERIFICATION_DIR "/c_01_raw_audio.bin", audio_data_orig, num_samples_orig);

    printf("\n[Step 2] Applying reflection padding...\n");
    size_t pad_samples = (size_t)(FRAME_PAD_MS / 1000.0 * SAMPLE_RATE);
    size_t num_samples_padded = num_samples_orig + 2 * pad_samples;
    float* audio_data_padded = malloc(num_samples_padded * sizeof(float));
    
    for (size_t i = 0; i < pad_samples; ++i) {
        audio_data_padded[pad_samples - 1 - i] = audio_data_orig[i + 1];
        audio_data_padded[pad_samples + num_samples_orig + i] = audio_data_orig[num_samples_orig - 2 - i];
    }
    memcpy(audio_data_padded + pad_samples, audio_data_orig, num_samples_orig * sizeof(float));
    drwav_free(audio_data_orig, NULL);

    printf("  -> Saving padded audio float data...\n");
    write_binary_file(VERIFICATION_DIR "/c_02_padded_audio.bin", audio_data_padded, num_samples_padded);

    printf("\n[Step 3] Computing full HCQT on padded audio...\n");
    size_t num_cqt_frames = 0;
    float* hcqt_data = NULL;
    for (int h = 0; h < N_HARMONICS; ++h) {
        printf("  - Harmonic %d/%d\n", h + 1, N_HARMONICS);
        CQTResult* cqt_res = compute_cqt(audio_data_padded, num_samples_padded, SAMPLE_RATE, HOP_LENGTH, FMIN * (h + 1), N_BINS, BINS_PER_OCTAVE);
        if (!cqt_res) { fprintf(stderr, "CQT computation failed.\n"); free(audio_data_padded); return 1; }
        if (h == 0) {
            num_cqt_frames = cqt_res->time_frames;
            hcqt_data = malloc(N_HARMONICS * N_BINS * num_cqt_frames * sizeof(float));
            if(!hcqt_data) { fprintf(stderr, "Failed to allocate HCQT buffer.\n"); return 1; }
        }
        for (size_t i = 0; i < cqt_res->freq_bins * num_cqt_frames; ++i) {
            hcqt_data[h * N_BINS * num_cqt_frames + i] = cabsf(cqt_res->data[i]);
        }
        free_cqt_result(cqt_res);
    }
    free(audio_data_padded);
    cleanup_cqt_caches();

    printf("  -> Saving full HCQT magnitude data...\n");
    write_binary_file(VERIFICATION_DIR "/c_03_full_hcqt.bin", hcqt_data, (long)N_HARMONICS * N_BINS * num_cqt_frames);

    printf("\n[Step 4] Extracting and saving each HCQT frame...\n");
    const float cqt_frame_duration_ms = (float)HOP_LENGTH / SAMPLE_RATE * 1000.0f;
    int frames_per_hop = (int)ceilf(INFERENCE_HOP_MS / cqt_frame_duration_ms);
    if (frames_per_hop < 1) frames_per_hop = 1;
    
    float* frame_buffer = (float*)aligned_malloc(N_HARMONICS * FRAME_WIDTH * N_BINS * sizeof(float), SIMD_ALIGNMENT);
    int frame_counter = 0;
    
    for (size_t f = 0; (f + FRAME_WIDTH) <= num_cqt_frames; f += frames_per_hop) {
        for (int h = 0; h < N_HARMONICS; ++h) {
            for (int t = 0; t < FRAME_WIDTH; ++t) {
                for (int b = 0; b < N_BINS; ++b) {
                    size_t t_in_full_cqt = f + t;
                    long src_index = (long)h * N_BINS * num_cqt_frames + (long)b * num_cqt_frames + t_in_full_cqt;
                    long dest_index = (long)h * FRAME_WIDTH * N_BINS + (long)t * N_BINS + b;
                    frame_buffer[dest_index] = hcqt_data[src_index];
                }
            }
        }
        char frame_filename[256];
        snprintf(frame_filename, sizeof(frame_filename), VERIFICATION_DIR "/c_frame_%03d.bin", frame_counter);
        write_binary_file(frame_filename, frame_buffer, N_HARMONICS * FRAME_WIDTH * N_BINS);
        frame_counter++;
    }
    
    printf("  -> Saved %d frames.\n", frame_counter);

    free(hcqt_data);
    aligned_free(frame_buffer);
    printf("\nSUCCESS: C verification data generated.\n");
    return 0;
}