#include "cpu_layers.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "utils.h"

#ifdef USE_NEON
#include <arm_neon.h>
#endif

#ifdef USE_SSE
#include <immintrin.h>
#endif

// Allocates memory aligned to <alignment> bytes so the NEON and SSE loads can use aligned addresses.
// Falls back to plain malloc if the aligned allocation fails.
void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return malloc(size);
    }
    return ptr;
}

// posix_memalign memory is released with a normal free.
void aligned_free(void* ptr) {
    free(ptr);
}

// Works out the sizes and strides once per layer so conv2d_cpu_opt does not repeat the multiplications for every output value.
void precompute_layer_opt_params(LayerOptParams* opt_params, const LayerParams* params) {
    opt_params->input_channel_size = params->h_in * params->w_in;
    opt_params->output_channel_size = params->h_out * params->w_out;
    opt_params->kernel_size = params->kernel_h * params->kernel_w;
    opt_params->total_output_size = params->n_out * params->c_out * params->h_out * params->w_out;
    opt_params->weight_channel_stride = params->c_in * params->kernel_h * params->kernel_w;
    opt_params->weight_kernel_stride = params->kernel_h * params->kernel_w;
}

// Direct 2D convolution of one sample, with the bias added to every output value.
// Steps for each output channel:
// 1. Split the output into tiles of 8 rows by 16 columns.
// 2. For each output value, sum input times weight over all input channels and kernel positions.
// 3. Skip kernel positions that fall in the zero padding.
// 4. Write the sum to the output.
// The tile size keeps the working set small enough for the L1 and L2 caches of the Jetson Nano.
// The output channel loop is the one split across threads, since each channel writes its own part of the output.
void conv2d_cpu_opt(const float* __restrict__ input,
                    float* __restrict__ output,
                    const LayerParams* __restrict__ params,
                    const float* __restrict__ weights,
                    const float* __restrict__ biases,
                    const LayerOptParams* __restrict__ opt_params) {
    
    const int h_out = params->h_out;
    const int w_out = params->w_out;
    const int h_in = params->h_in;
    const int w_in = params->w_in;
    const int c_in = params->c_in;
    const int c_out = params->c_out;
    const int kernel_h = params->kernel_h;
    const int kernel_w = params->kernel_w;
    const int pad_h = params->pad_h;
    const int pad_w = params->pad_w;
    const int stride_h = params->stride_h;
    const int stride_w = params->stride_w;
    
    // Tile size for the cache blocking described above the function.
    const int tile_h = 8;
    const int tile_w = 16;
    
    #pragma omp parallel for
    for (int oc = 0; oc < c_out; ++oc) {
        const float bias = (biases != NULL) ? biases[oc] : 0.0f;
        const float* __restrict__ weight_base = weights + (long)oc * opt_params->weight_channel_stride;
        
        for (int oh_tile = 0; oh_tile < h_out; oh_tile += tile_h) {
            const int oh_end = (oh_tile + tile_h < h_out) ? oh_tile + tile_h : h_out;
            
            for (int ow_tile = 0; ow_tile < w_out; ow_tile += tile_w) {
                const int ow_end = (ow_tile + tile_w < w_out) ? ow_tile + tile_w : w_out;
                
                for (int oh = oh_tile; oh < oh_end; ++oh) {
                    for (int ow = ow_tile; ow < ow_end; ++ow) {
                        float sum = bias;
                        
                        for (int ic = 0; ic < c_in; ++ic) {
                            const float* __restrict__ input_channel = input + (long)ic * opt_params->input_channel_size;
                            const float* __restrict__ weight_channel = weight_base + (long)ic * opt_params->weight_kernel_stride;
                            
                            for (int kh = 0; kh < kernel_h; ++kh) {
                                const int ih = oh * stride_h - pad_h + kh;
                                if (ih >= 0 && ih < h_in) {
                                    const float* __restrict__ input_row = input_channel + (long)ih * w_in;
                                    const float* __restrict__ weight_row = weight_channel + (long)kh * kernel_w;
                                    
                                    int kw = 0;
#ifdef USE_NEON
                                    // Handles 4 kernel columns at a time with fused multiply add.
                                    // A group that touches the padding is done one value at a time, because a vector load there would read outside the row.
                                    if (kernel_w >= 4) {
                                        float32x4_t sum_vec = vdupq_n_f32(0.0f);
                                        for (; kw <= kernel_w - 4; kw += 4) {
                                            int iw_base = ow * stride_w - pad_w + kw;
                                            if (iw_base >= 0 && iw_base + 3 < w_in) {
                                                float32x4_t input_vec = vld1q_f32(&input_row[iw_base]);
                                                float32x4_t weight_vec = vld1q_f32(&weight_row[kw]);
                                                sum_vec = vfmaq_f32(sum_vec, input_vec, weight_vec);
                                            } else {
                                                for (int k = 0; k < 4 && kw + k < kernel_w; ++k) {
                                                    int iw = iw_base + k;
                                                    if (iw >= 0 && iw < w_in) {
                                                        sum += input_row[iw] * weight_row[kw + k];
                                                    }
                                                }
                                                kw += 4;
                                                break;
                                            }
                                        }
                                        // Adds the 4 lanes together to finish the vector part of the sum.
                                        float sum_array[4];
                                        vst1q_f32(sum_array, sum_vec);
                                        sum += sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
                                    }
#endif
                                    // Handles the columns the vector loop did not cover, and the whole row when there is no NEON.
                                    for (; kw < kernel_w; ++kw) {
                                        const int iw = ow * stride_w - pad_w + kw;
                                        if (iw >= 0 && iw < w_in) {
                                            sum += input_row[iw] * weight_row[kw];
                                        }
                                    }
                                }
                            }
                        }
                        const long output_idx = (long)oc * opt_params->output_channel_size + (long)oh * w_out + ow;
                        output[output_idx] = sum;
                    }
                }
            }
        }
    }
}

// Leaky ReLU in place with a slope of 0.01 for negative values, which matches the PyTorch model.
// Uses vector instructions on groups of 4 values and a scalar loop for the remainder.
void leaky_relu_cpu_opt(float* __restrict__ tensor, const LayerParams* __restrict__ params) {
    const long total_elements = (long)params->n_out * params->c_out * params->h_out * params->w_out;
    const float leak_factor = 0.01f;
    
    long i = 0;
#ifdef USE_NEON
    float32x4_t zero_vec = vdupq_n_f32(0.0f);
    float32x4_t leak_vec = vdupq_n_f32(leak_factor);
    for (; i <= total_elements - 4; i += 4) {
        float32x4_t data_vec = vld1q_f32(&tensor[i]);
        uint32x4_t mask = vcltq_f32(data_vec, zero_vec);
        float32x4_t leak_data = vmulq_f32(data_vec, leak_vec);
        float32x4_t result = vbslq_f32(mask, leak_data, data_vec);
        vst1q_f32(&tensor[i], result);
    }
#elif defined(USE_SSE)
    __m128 zero_vec = _mm_setzero_ps();
    __m128 leak_vec = _mm_set1_ps(leak_factor);
    for (; i <= total_elements - 4; i += 4) {
        __m128 data_vec = _mm_load_ps(&tensor[i]);
        __m128 mask = _mm_cmplt_ps(data_vec, zero_vec);
        __m128 leak_data = _mm_mul_ps(data_vec, leak_vec);
        __m128 result = _mm_blendv_ps(data_vec, leak_data, mask);
        _mm_store_ps(&tensor[i], result);
    }
#endif
    for (; i < total_elements; ++i) {
        if (tensor[i] < 0.0f) {
            tensor[i] *= leak_factor;
        }
    }
}

// Sigmoid in place, used on the final layer to turn the logits into probabilities.
void sigmoid_cpu_opt(float* __restrict__ tensor, long num_elements) {
    for (long i = 0; i < num_elements; ++i) {
        tensor[i] = 1.0f / (1.0f + expf(-tensor[i]));
    }
}

// One convolution followed by leaky ReLU.
// The BatchNorm is already folded into the weights, so no separate normalisation step is needed.
void conv_relu_block(
    const float* input,
    float* output,
    const LayerParams* params,
    const LayerOptParams* opt_params,
    const ConvWithBias* weights
) {
    conv2d_cpu_opt(input, output, params, weights->weight, weights->bias, opt_params);
    leaky_relu_cpu_opt(output, params);
}

// Instance normalisation in place, done separately for each channel.
// Steps for each channel:
// 1. Compute the mean.
// 2. Compute the variance and then the standard deviation, with a small epsilon to avoid dividing by zero.
// 3. Normalise and apply the learned scale (gamma) and shift (beta).
// The sums use double to limit rounding error over a large number of values.
void instance_norm_cpu_opt(
    float* __restrict__ tensor,
    int num_channels,
    int channel_size,
    const InstanceNormWeights* __restrict__ weights
) {
    const float epsilon = 1e-5f;

    for (int c = 0; c < num_channels; ++c) {
        float* __restrict__ channel_data = tensor + (long)c * channel_size;
        
        double sum = 0.0;
        for (int i = 0; i < channel_size; ++i) {
            sum += channel_data[i];
        }
        const float mean = (float)(sum / channel_size);

        double sum_sq_diff = 0.0;
        for (int i = 0; i < channel_size; ++i) {
            const float diff = channel_data[i] - mean;
            sum_sq_diff += diff * diff;
        }
        const float std_dev = sqrtf((float)(sum_sq_diff / channel_size) + epsilon);
        const float inv_std_dev = 1.0f / std_dev;

        const float gamma = weights->gamma[c];
        const float beta = weights->beta[c];

        for (int i = 0; i < channel_size; ++i) {
            channel_data[i] = ((channel_data[i] - mean) * inv_std_dev) * gamma + beta;
        }
    }
}

// Frees the weight and bias arrays but not the struct itself.
void free_conv_with_bias(ConvWithBias* weights) {
    free(weights->weight);
    free(weights->bias);
}

// Frees the gamma and beta arrays but not the struct itself.
void free_instance_norm_weights(InstanceNormWeights* weights) {
    free(weights->gamma);
    free(weights->beta);
}