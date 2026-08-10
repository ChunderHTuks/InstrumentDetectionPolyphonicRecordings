#include <stdio.h>
#include "utils.h"
#include "weight_loader.h"

ConvWithBias load_folded_conv_block(const char* component, int layer_idx) {
    ConvWithBias block;
    char path[512];
    long dummy;
    
    snprintf(path, sizeof(path), "assets/c_weights/%s/%d_folded_weight.bin", component, layer_idx);
    block.weight = load_binary_file(path, &dummy);

    snprintf(path, sizeof(path), "assets/c_weights/%s/%d_folded_bias.bin", component, layer_idx);
    block.bias = load_binary_file(path, &dummy);
    
    return block;
}

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