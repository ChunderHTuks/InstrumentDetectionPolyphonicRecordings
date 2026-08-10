#ifndef CPU_LAYERS_H
#define CPU_LAYERS_H

#include <stdint.h>
#include <stddef.h>

#define SIMD_ALIGNMENT 16

typedef struct {
    int n_in, c_in, h_in, w_in;
    int k_out, kernel_h, kernel_w;
    int pad_h, pad_w, stride_h, stride_w;
    int n_out, c_out, h_out, w_out;
} __attribute__((aligned(SIMD_ALIGNMENT))) LayerParams;

typedef struct {
    int input_channel_size;
    int output_channel_size;
    int kernel_size;
    int total_output_size;
    int weight_channel_stride;
    int weight_kernel_stride;
} LayerOptParams;

typedef struct {
    float* gamma;
    float* beta;
} InstanceNormWeights;

typedef struct {
    float* weight;
    float* bias;
} ConvWithBias;

#ifdef __cplusplus
extern "C" {
#endif

void conv2d_cpu_opt(const float* __restrict__ input,
                    float* __restrict__ output,
                    const LayerParams* __restrict__ params,
                    const float* __restrict__ weights,
                    const float* __restrict__ biases,
                    const LayerOptParams* __restrict__ opt_params);

void leaky_relu_cpu_opt(float* __restrict__ tensor, const LayerParams* __restrict__ params);

void sigmoid_cpu_opt(float* __restrict__ tensor, long num_elements);

void conv_relu_block(
    const float* input,
    float* output,
    const LayerParams* params,
    const LayerOptParams* opt_params,
    const ConvWithBias* weights
);

void instance_norm_cpu_opt(
    float* __restrict__ tensor,
    int num_channels,
    int channel_size,
    const InstanceNormWeights* __restrict__ weights
);

void precompute_layer_opt_params(LayerOptParams* opt_params, const LayerParams* params);
void* aligned_malloc(size_t size, size_t alignment);
void aligned_free(void* ptr);

void free_conv_with_bias(ConvWithBias* weights);
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