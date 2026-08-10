#ifndef WEIGHT_LOADER_H
#define WEIGHT_LOADER_H

#include "cpu_layers.h" // Needs ConvWithBias and InstanceNormWeights types

// --- Function Declarations ---

ConvWithBias load_folded_conv_block(const char* component, int layer_idx);

ConvWithBias load_conv_with_bias(const char* component, const char* name);

InstanceNormWeights load_instance_norm_weights(void); // Use void for clarity

#endif // WEIGHT_LOADER_H