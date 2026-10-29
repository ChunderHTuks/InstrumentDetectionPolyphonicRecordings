#include <fftw3.h>
#include "cqt.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <omp.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Cache of Hanning windows, one per window length.
// Every CQT bin has its own window length, and some lengths repeat, so each one is built only once.
typedef struct {
    size_t length;
    float* window;
} WindowCache;
static WindowCache* window_cache = NULL;
static size_t window_cache_size = 0;
static size_t window_cache_capacity = 0;

// Cache of the frequency domain kernels, kept between calls to compute_cqt.
// The kernels only depend on the fields below, so a later call with the same settings skips the expensive setup.
typedef struct {
    size_t fft_size;
    int bins_per_octave;
    float fmin;
    int n_kernels;
    float complex** kernels;
    int valid;
} KernelCache;
static KernelCache kernel_cache = {0};

static int can_use_cached_kernels(size_t fft_size, int bins_per_octave, float fmin, int n_bins);
static void cleanup_kernel_cache(void);
static void free_resources(CQTResult* result, float* padded_audio, float complex** kernels,
                          int kernel_count, fftwf_plan plan, float complex* temp_buffer);

// Returns the Hanning window for this length, building and caching it on the first request.
// The cache array doubles in size when it is full.
static const float* get_hanning_window(size_t length) {
    for (size_t i = 0; i < window_cache_size; i++) {
        if (window_cache[i].length == length)
          return window_cache[i].window;
    }
    if (window_cache_size >= window_cache_capacity) {
        window_cache_capacity = window_cache_capacity == 0 ? 8 : window_cache_capacity * 2;
        window_cache = realloc(window_cache, window_cache_capacity * sizeof(WindowCache));
        if (!window_cache) {
            fprintf(stderr, "Failed to allocate window cache\n");
            return NULL;
        }
    }
    float* window = malloc(length * sizeof(float));
    if (!window) {
        fprintf(stderr, "Failed to allocate window\n");
        return NULL;
    }
    float factor = 2.0f * M_PI / (length - 1);
    for (size_t i = 0; i < length; i++) {
        window[i] = 0.5f * (1.0f - cosf(factor * i));
    }
    window_cache[window_cache_size].length = length;
    window_cache[window_cache_size].window = window;
    window_cache_size++;
    return window;
}

// Rounds up to the next power of 2, using the bit smearing trick.
// FFTW is fastest on power of 2 sizes.
static size_t next_power_of_2(size_t n) {
    if (n <= 1) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    if (sizeof(size_t) > 4) n |= n >> 32;
    return ++n;
}

// Builds the time domain kernel for one CQT bin.
// Steps:
// 1. The window length is Q cycles of this frequency, so low bins get long windows and high bins get short ones.
// 2. Multiply the Hanning window by a complex sinusoid at the bin frequency, centred on the window middle.
// 3. Normalise so the kernel has unit energy, which keeps the bins on the same scale.
// 4. Copy it into the middle of a zero filled array of length fft_size.
// A bin whose window does not fit in fft_size gets an all zero kernel.
static float complex* generate_cqt_kernel(float frequency, float sample_rate, float Q, size_t fft_size) {
    int window_length = (int)roundf(Q * sample_rate / frequency);
    if (window_length <= 0 || window_length > (int)fft_size) {
        float complex* kernel = fftwf_malloc(sizeof(float complex) * fft_size);
        if (kernel) memset(kernel, 0, sizeof(float complex) * fft_size);
        return kernel;
    }
    const float* window = get_hanning_window(window_length);
    if (!window) return NULL;
    float complex* temp_kernel = fftwf_malloc(sizeof(float complex) * window_length);
    if (!temp_kernel) return NULL;
    int half_window = window_length / 2;
    float phase_factor = 2.0f * M_PI * frequency / sample_rate;
    for (int i = 0; i < window_length; i++) {
        float phase = phase_factor * (i - half_window);
        temp_kernel[i] = window[i] * (cosf(phase) + sinf(phase) * I);
    }
    float norm = 0.0f;
    for (int i = 0; i < window_length; i++) {
        norm += crealf(temp_kernel[i]) * crealf(temp_kernel[i]) + cimagf(temp_kernel[i]) * cimagf(temp_kernel[i]);
    }
    if (norm > 0.0f) {
        float inv_norm = 1.0f / sqrtf(norm);
        for (int i = 0; i < window_length; i++) temp_kernel[i] *= inv_norm;
    }
    float complex* kernel = fftwf_malloc(sizeof(float complex) * fft_size);
    if (!kernel) { fftwf_free(temp_kernel); return NULL; }
    memset(kernel, 0, sizeof(float complex) * fft_size);
    int start = ((int)fft_size - window_length) / 2;
    if (start >= 0 && start + window_length <= (int)fft_size) {
        memcpy(kernel + start, temp_kernel, sizeof(float complex) * window_length);
    }
    fftwf_free(temp_kernel);
    return kernel;
}

// Computes the CQT with the direct method.
// Each frame is transformed with one FFT and then multiplied with the spectral kernel of every bin.
// Steps:
// 1. Load the FFTW wisdom file if it exists, so the plans do not need to be measured again.
// 2. Work out Q, and choose an FFT size that fits the longest kernel (at least 4096).
// 3. Pad the audio on both sides by half an FFT size, mirrored around the first and last samples, so the first and last frames are centred on the audio edges.
// 4. Build the kernels, or reuse the cached ones.
// 5. For every frame, run the FFT and take the dot product with each conjugated kernel. Frames are split across threads.
// 6. Save the wisdom file and free the temporary buffers.
// Bins at or above the Nyquist frequency are dropped, so freq_bins in the result can be smaller than n_bins.
// The result is laid out as [bin][frame] and the caller frees it with free_cqt_result.
CQTResult* compute_cqt(const float* audio, size_t audio_length, float sample_rate,
                      int hop_length, float fmin, int n_bins, int bins_per_octave) {
    if (!audio || audio_length == 0 || n_bins <= 0) return NULL;

    const char* wisdom_file = "fftw_wisdom.dat";
    if (fftwf_import_wisdom_from_filename(wisdom_file) != 0) {
        printf(" -> FFTW wisdom loaded from %s\n", wisdom_file);
    } else {
        printf(" -> No FFTW wisdom file found. A new one will be generated.\n");
        printf("    (This first run will be slower as FFTW optimizes for your hardware)\n");
    }
    // FFTW_MEASURE tries several algorithms to find the fastest on this hardware, which is why the first run is slow.
    const unsigned FFTW_PLAN_FLAG = FFTW_MEASURE;

    int total_bins = n_bins;
    // Q is the ratio of the centre frequency to the bandwidth, which is the same for every bin.
    float Q = 1.0f / (powf(2.0f, 1.0f / bins_per_octave) - 1.0f);
    size_t min_kernel_len = (size_t)ceilf(Q * sample_rate / fmin);
    size_t fft_size = next_power_of_2(min_kernel_len > 4096 ? min_kernel_len : 4096);
    size_t padding = fft_size / 2;
    size_t padded_length = audio_length + 2 * padding;

    float* padded_audio = malloc(padded_length * sizeof(float));
    if (!padded_audio) return NULL;

    // Mirrors the audio into the padding on both sides.
    for (size_t i = 0; i < padding; i++) {
        padded_audio[padding - 1 - i] = audio[i + 1];
        padded_audio[padding + audio_length + i] = audio[audio_length - 2 - i];
    }
    memcpy(padded_audio + padding, audio, audio_length * sizeof(float));

    int num_frames = (padded_length > fft_size) ? ((padded_length - fft_size) / hop_length + 1) : 0;
    if (num_frames <= 0) { free(padded_audio); return NULL; }

    CQTResult* result = malloc(sizeof(CQTResult));
    if (!result) { free(padded_audio); return NULL; }
    result->data = fftwf_malloc(sizeof(float complex) * total_bins * num_frames);
    if (!result->data) { free(result); free(padded_audio); return NULL; }
    result->freq_bins = total_bins;
    result->time_frames = num_frames;

    float complex** kernels;
    if (can_use_cached_kernels(fft_size, bins_per_octave, fmin, total_bins)) {
        kernels = kernel_cache.kernels;
    } else {
        cleanup_kernel_cache();
        kernels = malloc(total_bins * sizeof(float complex*));
        if (!kernels) {
            free_resources(result, padded_audio, NULL, 0, NULL, NULL);
            return NULL;
        }
        // Each kernel is moved to the frequency domain once here, so the frame loop only needs a multiplication.
        float complex* temp_kernel_fft = fftwf_malloc(sizeof(float complex) * fft_size);
        fftwf_plan kernel_plan = fftwf_plan_dft_1d(fft_size, (fftwf_complex*)temp_kernel_fft, (fftwf_complex*)temp_kernel_fft, FFTW_FORWARD, FFTW_PLAN_FLAG);
        float fmax = sample_rate / 2.0f;
        for (int bin = 0; bin < total_bins; bin++) {
            float frequency = fmin * powf(2.0f, (float)bin / bins_per_octave);
            if (frequency >= fmax) { result->freq_bins = bin; total_bins = bin; break; }
            kernels[bin] = generate_cqt_kernel(frequency, sample_rate, Q, fft_size);
            if (!kernels[bin]) { free_resources(result, padded_audio, kernels, bin, kernel_plan, temp_kernel_fft); return NULL; }
            memcpy(temp_kernel_fft, kernels[bin], sizeof(float complex) * fft_size);
            fftwf_execute(kernel_plan);
            memcpy(kernels[bin], temp_kernel_fft, sizeof(float complex) * fft_size);
        }
        fftwf_destroy_plan(kernel_plan);
        fftwf_free(temp_kernel_fft);
        kernel_cache.fft_size = fft_size; kernel_cache.bins_per_octave = bins_per_octave;
        kernel_cache.fmin = fmin; kernel_cache.n_kernels = total_bins;
        kernel_cache.kernels = kernels; kernel_cache.valid = 1;
    }

    fftwf_complex* frame_td = fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    fftwf_complex* frame_fd = fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    fftwf_plan forward_plan = fftwf_plan_dft_1d(fft_size, frame_td, frame_fd, FFTW_FORWARD, FFTW_PLAN_FLAG);
    float inv_fft_size = 1.0f / fft_size;

    // Each thread gets its own buffers, because the threads work on different frames at the same time.
    int max_threads = omp_get_max_threads();
    float complex** private_td_buffers = malloc(max_threads * sizeof(float complex*));
    float complex** private_fd_buffers = malloc(max_threads * sizeof(float complex*));
    for(int i=0; i<max_threads; ++i) {
        private_td_buffers[i] = fftwf_malloc(sizeof(float complex) * fft_size);
        private_fd_buffers[i] = fftwf_malloc(sizeof(float complex) * fft_size);
    }

    #pragma omp parallel for
    for (int frame = 0; frame < num_frames; frame++) {
        int thread_id = omp_get_thread_num();
        float complex* private_frame_td = private_td_buffers[thread_id];
        float complex* private_frame_fd = private_fd_buffers[thread_id];
        
        int start_sample = frame * hop_length;
        for (size_t i = 0; i < fft_size; i++) {
            private_frame_td[i] = padded_audio[start_sample + i];
        }

        fftwf_execute_dft(forward_plan, (fftwf_complex*)private_frame_td, (fftwf_complex*)private_frame_fd);

        for (int bin = 0; bin < total_bins; bin++) {
            float complex total_product = 0.0f + 0.0f * I;
            #ifdef __aarch64__
                // Computes the sum of frame * conj(kernel) for 4 complex values at a time.
                // vld2q_f32 splits the real and imaginary parts into separate vectors.
                float32x4_t sum_re = vdupq_n_f32(0.0f);
                float32x4_t sum_im = vdupq_n_f32(0.0f);
                for (size_t i = 0; i < fft_size; i += 4) {
                    float32x4x2_t frame_vec = vld2q_f32((float*)&private_frame_fd[i]);
                    float32x4x2_t kernel_vec = vld2q_f32((float*)&kernels[bin][i]);
                    sum_re = vmlaq_f32(sum_re, frame_vec.val[0], kernel_vec.val[0]);
                    sum_re = vmlaq_f32(sum_re, frame_vec.val[1], kernel_vec.val[1]);
                    sum_im = vmlaq_f32(sum_im, frame_vec.val[1], kernel_vec.val[0]);
                    sum_im = vmlsq_f32(sum_im, frame_vec.val[0], kernel_vec.val[1]);
                }
                float total_re = vaddvq_f32(sum_re);
                float total_im = vaddvq_f32(sum_im);
                total_product = total_re + total_im * I;
            #else
                for (size_t i = 0; i < fft_size; i++) {
                    total_product += private_frame_fd[i] * conjf(kernels[bin][i]);
                }
            #endif
            result->data[bin * num_frames + frame] = total_product * inv_fft_size;
        }
    }
    
    // Saved after the plans exist, so the next run can load them.
    fftwf_export_wisdom_to_filename(wisdom_file);

    for(int i=0; i<max_threads; ++i) {
        fftwf_free(private_td_buffers[i]);
        fftwf_free(private_fd_buffers[i]);
    }
    free(private_td_buffers);
    free(private_fd_buffers);

    fftwf_destroy_plan(forward_plan);
    fftwf_free(frame_td);
    fftwf_free(frame_fd);
    free(padded_audio);

    return result;
}

// Frees the data array and the struct. Safe to call with NULL.
void free_cqt_result(CQTResult* result) {
    if (result) {
        if (result->data) fftwf_free(result->data);
        free(result);
    }
}

// Frees the window cache and the kernel cache and releases the FFTW internal data.
// Call once at the end of the program.
void cleanup_cqt_caches(void) {
    for (size_t i = 0; i < window_cache_size; i++) {
        if (window_cache[i].window) free(window_cache[i].window);
    }
    if (window_cache) {
        free(window_cache);
        window_cache = NULL;
    }
    window_cache_size = 0;
    window_cache_capacity = 0;
    cleanup_kernel_cache();
    fftwf_cleanup();
}

// The cached kernels can be reused only if they were built with the same settings and cover enough bins.
static int can_use_cached_kernels(size_t fft_size, int bins_per_octave, float fmin, int n_bins) {
    return kernel_cache.valid && kernel_cache.fft_size == fft_size &&
           kernel_cache.bins_per_octave == bins_per_octave &&
           fabsf(kernel_cache.fmin - fmin) < 1e-6f &&
           kernel_cache.n_kernels >= n_bins;
}

// Frees every cached kernel and marks the cache as empty.
static void cleanup_kernel_cache(void) {
    if (kernel_cache.kernels) {
        for (int i = 0; i < kernel_cache.n_kernels; i++) {
            if (kernel_cache.kernels[i]) fftwf_free(kernel_cache.kernels[i]);
        }
        free(kernel_cache.kernels);
        kernel_cache.kernels = NULL;
    }
    kernel_cache.valid = 0;
    kernel_cache.n_kernels = 0;
}

// Frees whatever has been allocated so far when compute_cqt has to stop early.
// Any argument can be NULL if that resource does not exist yet.
static void free_resources(CQTResult* result, float* padded_audio, float complex** kernels,
                          int kernel_count, fftwf_plan plan, float complex* temp_buffer) {
    if (plan) fftwf_destroy_plan(plan);
    if (temp_buffer) fftwf_free(temp_buffer);
    if (kernels) {
        for (int i = 0; i < kernel_count; i++) {
            if (kernels[i]) fftwf_free(kernels[i]);
        }
        free(kernels);
    }
    if (result) {
        if (result->data)
          fftwf_free(result->data);
        free(result);
    }
    if (padded_audio)
      free(padded_audio);
}