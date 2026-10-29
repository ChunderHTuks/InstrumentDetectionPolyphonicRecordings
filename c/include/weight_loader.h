#ifndef WEIGHT_LOADER_H
#define WEIGHT_LOADER_H

#include "cpu_layers.h" // Needs ConvWithBias and InstanceNormWeights types

// --- Function Declarations ---

// Loads the folded weight and bias files of one backbone or head layer.
ConvWithBias load_folded_conv_block(const char* component, int layer_idx);

// Loads the un-folded weight and bias files of a named layer, such as a classifier.
ConvWithBias load_conv_with_bias(const char* component, const char* name);

// Loads the gamma and beta of the input instance normalisation.
InstanceNormWeights load_instance_norm_weights(void); // Use void for clarity

#endif // WEIGHT_LOADER_H