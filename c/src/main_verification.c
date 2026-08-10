#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "cpu_layers.h"
#include "weight_loader.h"
#include "model_params.h"

// --- Configuration ---
#define VERIFICATION_DIR "./verification_data"
#define PYTHON_INPUT_FILE "./verification_data/py_01_input.bin"

// --- Global State for Saving Files ---
int C_FILE_COUNTER = 1;

// --- Helper function to save tensors with correct naming ---
void save_c_tensor(const float* data, long num_elements, const char* name) {
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "%s/c_%02d_%s.bin", VERIFICATION_DIR, C_FILE_COUNTER, name);
    printf("  -> Saving C tensor to %s\n", filepath);
    write_binary_file(filepath, data, num_elements);
    C_FILE_COUNTER++;
}

int main() {
    printf("--- C Implementation Verification Data Generator ---\n");

    // --- 1. Load Input ---
    printf("\n[Step 1] Loading Python-generated input tensor...\n");
    long input_size;
    float* h_input = load_binary_file(PYTHON_INPUT_FILE, &input_size);
    if (!h_input) { /* ... error handling ... */ return 1; }
    save_c_tensor(h_input, input_size, "input");

    // --- 2. Load Weights ---
    printf("\n[Step 2] Loading folded model weights...\n");
    InstanceNormWeights norm_weights = load_instance_norm_weights();
    ConvWithBias backbone_weights[NUM_BACKBONE_LAYERS];
     int pytorch_layer_indices[] = {0, 3, 6, 9, 12, 15};
    for (int i = 0; i < NUM_BACKBONE_LAYERS; ++i) {
        // Use the correct index from the array, not the loop variable `i`
        backbone_weights[i] = load_folded_conv_block("backbone", pytorch_layer_indices[i]);
    }
    // ======================== END: CORRECTED BLOCK ========================

    ConvWithBias piano_agg_weights = load_folded_conv_block("piano", 0);
    ConvWithBias piano_cls_weights = load_conv_with_bias("piano", "classifier");
    ConvWithBias strings_agg_weights = load_folded_conv_block("strings", 0);
    ConvWithBias strings_cls_weights = load_conv_with_bias("strings", "classifier");
    ConvWithBias woodwinds_agg_weights = load_folded_conv_block("woodwinds", 0);
    ConvWithBias woodwinds_cls_weights = load_conv_with_bias("woodwinds", "classifier");
    ConvWithBias brass_agg_weights = load_folded_conv_block("brass", 0);
    ConvWithBias brass_cls_weights = load_conv_with_bias("brass", "classifier");
    // --- 3. Pre-compute Layer Params ---
    printf("\n[Step 3] Pre-computing layer optimization parameters...\n");
    LayerOptParams backbone_opt_params[NUM_BACKBONE_LAYERS];
    for(int i=0; i<NUM_BACKBONE_LAYERS; ++i) precompute_layer_opt_params(&backbone_opt_params[i], &backbone_architecture[i]);
    LayerOptParams head_opt_params[4][2];
    precompute_layer_opt_params(&head_opt_params[0][0], &piano_head_architecture[0]); precompute_layer_opt_params(&head_opt_params[0][1], &piano_head_architecture[1]);
    precompute_layer_opt_params(&head_opt_params[1][0], &strings_head_architecture[0]); precompute_layer_opt_params(&head_opt_params[1][1], &strings_head_architecture[1]);
    precompute_layer_opt_params(&head_opt_params[2][0], &woodwinds_head_architecture[0]); precompute_layer_opt_params(&head_opt_params[2][1], &woodwinds_head_architecture[1]);
    precompute_layer_opt_params(&head_opt_params[3][0], &brass_head_architecture[0]); precompute_layer_opt_params(&head_opt_params[3][1], &brass_head_architecture[1]);

    // --- 4. Allocate Intermediate Buffers ---
    printf("\n[Step 4] Allocating intermediate memory buffers...\n");
    long max_buffer_elements = 32 * 44 * 84;
    float* bufferA = (float*)aligned_malloc(max_buffer_elements * sizeof(float), SIMD_ALIGNMENT);
    float* bufferB = (float*)aligned_malloc(max_buffer_elements * sizeof(float), SIMD_ALIGNMENT);
    if (!bufferA || !bufferB) { fprintf(stderr, "Failed to allocate ping-pong buffers\n"); return 1; }
    memcpy(bufferA, h_input, input_size * sizeof(float));
    free(h_input);

    const LayerParams* piano_agg_layer = &piano_head_architecture[0];
    long agg_out_size = (long)piano_agg_layer->c_out * piano_agg_layer->h_out * piano_agg_layer->w_out;
    float* agg_out_buffer = (float*)aligned_malloc(agg_out_size * sizeof(float), SIMD_ALIGNMENT);
    
    // --- 5. Run Step-by-Step Inference and Save ---
    printf("\n[Step 5] Running model and saving intermediate C tensors...\n");

    const LayerParams* first_layer = &backbone_architecture[0];
    instance_norm_cpu_opt(bufferA, first_layer->c_in, first_layer->h_in * first_layer->w_in, &norm_weights);
    save_c_tensor(bufferA, input_size, "input_norm");

    const float* current_input = bufferA;
    float* current_output = bufferB;
    for (int i = 0; i < NUM_BACKBONE_LAYERS; ++i) {
        if (i % 2 == 0) { current_input = bufferA; current_output = bufferB; } 
        else { current_input = bufferB; current_output = bufferA; }
        conv_relu_block(current_input, current_output, &backbone_architecture[i], &backbone_opt_params[i], &backbone_weights[i]);
        char name[50];
        sprintf(name, "backbone_%d_conv_relu", i);
        save_c_tensor(current_output, backbone_opt_params[i].total_output_size, name);
    }
    const float* backbone_out_buffer = current_output;

    // --- Heads ---
    const char* heads[] = {"piano", "strings", "woodwinds", "brass"};
    const ConvWithBias* aggs[] = {&piano_agg_weights, &strings_agg_weights, &woodwinds_agg_weights, &brass_agg_weights};
    const ConvWithBias* clss[] = {&piano_cls_weights, &strings_cls_weights, &woodwinds_cls_weights, &brass_cls_weights};
    const LayerParams* head_archs[][2] = {
        {&piano_head_architecture[0], &piano_head_architecture[1]},
        {&strings_head_architecture[0], &strings_head_architecture[1]},
        {&woodwinds_head_architecture[0], &woodwinds_head_architecture[1]},
        {&brass_head_architecture[0], &brass_head_architecture[1]}
    };

    for (int i = 0; i < 4; ++i) {
        long head_out_size = head_opt_params[i][1].total_output_size;
        float* head_out_buffer = (float*)aligned_malloc(head_out_size * sizeof(float), SIMD_ALIGNMENT);
        
        // ======================= START: CORRECTED BLOCK =======================
        // Aggregator
        conv_relu_block(backbone_out_buffer, agg_out_buffer, head_archs[i][0], &head_opt_params[i][0], aggs[i]);
        char agg_name[50]; sprintf(agg_name, "%s_agg_conv_relu", heads[i]);
        save_c_tensor(agg_out_buffer, head_opt_params[i][0].total_output_size, agg_name);
        
        // Classifier
        conv2d_cpu_opt(agg_out_buffer, head_out_buffer, head_archs[i][1], clss[i]->weight, clss[i]->bias, &head_opt_params[i][1]);
        char cls_name[50]; sprintf(cls_name, "%s_classifier_conv", heads[i]);
        save_c_tensor(head_out_buffer, head_out_size, cls_name);
        // ======================== END: CORRECTED BLOCK ========================
        
        // Sigmoid
        sigmoid_cpu_opt(head_out_buffer, head_out_size);
        char sig_name[50]; sprintf(sig_name, "%s_sigmoid", heads[i]);
        save_c_tensor(head_out_buffer, head_out_size, sig_name);
        
        aligned_free(head_out_buffer);
    }
    aligned_free(agg_out_buffer);

    // --- 6. Cleanup ---
    printf("\n[Step 6] Releasing all allocated resources...\n");
    aligned_free(bufferA);
    aligned_free(bufferB);
    
    free_instance_norm_weights(&norm_weights);
    for (int i = 0; i < NUM_BACKBONE_LAYERS; ++i) free_conv_with_bias(&backbone_weights[i]);
    free_conv_with_bias(&piano_agg_weights); free_conv_with_bias(&piano_cls_weights);
    free_conv_with_bias(&strings_agg_weights); free_conv_with_bias(&strings_cls_weights);
    free_conv_with_bias(&woodwinds_agg_weights); free_conv_with_bias(&woodwinds_cls_weights);
    free_conv_with_bias(&brass_agg_weights); free_conv_with_bias(&brass_cls_weights);

    printf("\nSUCCESS: Generated %d C verification files in '%s'.\n", C_FILE_COUNTER - 1, VERIFICATION_DIR);
    printf("You can now copy the 'verification_data' folder back to your PC and run 'verification.py'.\n");
    return 0;
}