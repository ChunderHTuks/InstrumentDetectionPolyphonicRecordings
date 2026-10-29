// Loads the exported model weights from assets/c_weights/ into float arrays.
#include <stdio.h>
#include "utils.h"
#include "weight_loader.h"

// Loads <layer_idx>_folded_weight.bin and <layer_idx>_folded_bias.bin for one layer.
// These are the Conv2d weights with the BatchNorm already folded in, so only a plain conv plus bias is needed.
ConvWithBias load_folded_conv_block(const char* component, int layer_idx) {
    ConvWithBias block;
    char path[512];
    long dummy; // Element count is not needed, the layer shapes come from model_params.h
    
    snprintf(path, sizeof(path), "assets/c_weights/%s/%d_folded_weight.bin", component, layer_idx);
    block.weight = load_binary_file(path, &dummy);

    snprintf(path, sizeof(path), "assets/c_weights/%s/%d_folded_bias.bin", component, layer_idx);
    block.bias = load_binary_file(path, &dummy);
    
    return block;
}

// Loads <name>_weight.bin and <name>_bias.bin for a layer that was not folded, such as a classifier.
ConvWithBias load_conv_with_bias(const char* component, const char* name) {
    ConvWithBias weights;
    char path[512];
    long dummy;
    snprintf(path, sizeof(path), "assets/c_weights/%s/%s_weight.bin", component, name);
    weights.weight = load_binary_file(path, &dummy);
    
    snprintf(path, sizeof(path), "assets/c_weights/%s/%s_bias.bin", component, name);
    weights.bias = load_binary_file(path, &dummy);
    
    return weights;
}

// Loads the gamma (weight.bin) and beta (bias.bin) of the input instance normalisation.
InstanceNormWeights load_instance_norm_weights(void) {
    InstanceNormWeights weights;
    char path[512];
    long dummy;
    snprintf(path, sizeof(path), "assets/c_weights/input_norm/weight.bin");
    weights.gamma = load_binary_file(path, &dummy);

    snprintf(path, sizeof(path), "assets/c_weights/input_norm/bias.bin");
    weights.beta = load_binary_file(path, &dummy);

    return weights;
}