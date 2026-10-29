#ifndef CPU_LAYERS_H
#define CPU_LAYERS_H

#include <stdint.h>
#include <stddef.h>

// Memory alignment in bytes. 16 matches the 128-bit NEON and SSE vector width.
#define SIMD_ALIGNMENT 16

// Shape, kernel, padding and stride settings for one convolution layer, plus its output shape.
typedef struct {
    int n_in, c_in, h_in, w_in;
    int k_out, kernel_h, kernel_w;
    int pad_h, pad_w, stride_h, stride_w;
    int n_out, c_out, h_out, w_out;
} __attribute__((aligned(SIMD_ALIGNMENT))) LayerParams;

// Sizes derived from LayerParams once per layer, so conv2d_cpu_opt does not recompute them per pixel.
typedef struct {
    int input_channel_size;
    int output_channel_size;
    int kernel_size;
    int total_output_size;
    int weight_channel_stride;
    int weight_kernel_stride;
} LayerOptParams;

// Learnable scale (gamma) and shift (beta) of the input instance normalisation, one value per channel.
typedef struct {
    float* gamma;
    float* beta;
} InstanceNormWeights;

// Convolution weights and bias. The BatchNorm is already folded into these.
typedef struct {
    float* weight;
    float* bias;
} ConvWithBias;

#ifdef __cplusplus
extern "C" {
#endif

// Tiled, OpenMP-parallel, NEON-vectorised 2D convolution with bias.
void conv2d_cpu_opt(const float* __restrict__ input,
                    float* __restrict__ output,
                    const LayerParams* __restrict__ params,
                    const float* __restrict__ weights,
                    const float* __restrict__ biases,
                    const LayerOptParams* __restrict__ opt_params);

// Applies leaky ReLU in place over the whole output tensor of the layer.
void leaky_relu_cpu_opt(float* __restrict__ tensor, const LayerParams* __restrict__ params);

// Applies the sigmoid in place to num_elements floats.
void sigmoid_cpu_opt(float* __restrict__ tensor, long num_elements);

// Runs a convolution followed by leaky ReLU on the same output buffer.
void conv_relu_block(
    const float* input,
    float* output,
    const LayerParams* params,
    const LayerOptParams* opt_params,
    const ConvWithBias* weights
);

// Normalises each channel to zero mean and unit variance in place, then applies gamma and beta.
void instance_norm_cpu_opt(
    float* __restrict__ tensor,
    int num_channels,
    int channel_size,
    const InstanceNormWeights* __restrict__ weights
);

// Fills opt_params with the sizes and strides that conv2d_cpu_opt needs for this layer.
void precompute_layer_opt_params(LayerOptParams* opt_params, const LayerParams* params);
// Allocates size bytes aligned to the given alignment. Free with aligned_free.
void* aligned_malloc(size_t size, size_t alignment);
// Frees memory from aligned_malloc.
void aligned_free(void* ptr);

// Frees the weight and bias arrays of a ConvWithBias.
void free_conv_with_bias(ConvWithBias* weights);
// Frees the gamma and beta arrays.
void free_instance_norm_weights(InstanceNormWeights* weights);

#ifdef USE_NEON
#include <arm_neon.h>
#endif

#ifdef USE_SSE
#include <immintrin.h>
#endif

#ifdef __cplusplus
}
#endif

#endif // CPU_LAYERS_H