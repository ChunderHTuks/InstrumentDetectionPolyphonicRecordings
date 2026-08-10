#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"
#include "cpu_layers.h"
#include "weight_loader.h"
#include "model_params.h"

// --- Configuration ---
#define C_HCQT_FILE "./verification_data/c_03_full_hcqt.bin"
#define C_NORM_OUT_FILE "./verification_data/c_04_instancenorm_output.bin"
#define N_HARMONICS 5
#define N_BINS 84

int main() {
    printf("--- C InstanceNorm Verification ---\n");

    printf("\n[Step 1] Loading C-generated HCQT data from '%s'...\n", C_HCQT_FILE);
    long num_elements;
    float* hcqt_data = load_binary_file(C_HCQT_FILE, &num_elements);
    if (!hcqt_data) {
        fprintf(stderr, "FATAL: Could not load the C HCQT file. Please run 'dataprep_verifier' first.\n");
        return 1;
    }

    long channel_size = num_elements / N_HARMONICS;
    printf("  - Loaded %ld elements. Each of the %d channels has %ld elements.\n", num_elements, N_HARMONICS, channel_size);

    printf("\n[Step 2] Loading InstanceNorm weights...\n");
    InstanceNormWeights norm_weights = load_instance_norm_weights();

    printf("\n[Step 3] Calculating InstanceNorm and statistics channel by channel...\n");
    const float epsilon = 1e-5f;

    for (int c = 0; c < N_HARMONICS; ++c) {
        float* channel_data = hcqt_data + (long)c * channel_size;
        
        double sum = 0.0;
        for (long i = 0; i < channel_size; ++i) {
            sum += channel_data[i];
        }
        float mean = (float)(sum / channel_size);

        double sum_sq_diff = 0.0;
        for (long i = 0; i < channel_size; ++i) {
            const float diff = channel_data[i] - mean;
            sum_sq_diff += diff * diff;
        }
        float std_dev = sqrtf((float)(sum_sq_diff / channel_size) + epsilon);
        
        printf("\n  --- Channel %d (Harmonic %d) ---\n", c, c + 1);
        printf("  C Calculated Mean:   %.8f\n", mean);
        printf("  C Calculated StdDev: %.8f\n", std_dev);

        const float inv_std_dev = 1.0f / std_dev;
        const float gamma = norm_weights.gamma[c];
        const float beta = norm_weights.beta[c];
        
        #pragma omp parallel for
        for (long i = 0; i < channel_size; ++i) {
            channel_data[i] = (((channel_data[i] - mean) * inv_std_dev) * gamma) + beta;
        }
    }

    printf("\n[Step 4] Saving final normalized output to '%s'...\n", C_NORM_OUT_FILE);
    write_binary_file(C_NORM_OUT_FILE, hcqt_data, num_elements);

    free(hcqt_data);
    free_instance_norm_weights(&norm_weights);

    printf("\nSUCCESS: C InstanceNorm verification data generated.\n");
    return 0;
}