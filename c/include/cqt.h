#ifndef CQT_PROJECT_CQT_H
#define CQT_PROJECT_CQT_H

#include <stddef.h>

// This block creates a common type "cqt_complex" that works in both C and C++
#ifdef __cplusplus
// This code is seen only by the C++ compiler
#include <complex>
typedef std::complex<float> cqt_complex;
#else
// This code is seen only by the C compiler
#include <complex.h>
typedef float complex cqt_complex;
#endif


// Output of compute_cqt: a row-major bins x frames complex matrix.
// The struct now uses our common, language-agnostic type
typedef struct {
    cqt_complex* data;
    size_t freq_bins;
    size_t time_frames;
} CQTResult;

// The 'extern "C"' block allows C++ to link to C functions
#ifdef __cplusplus
extern "C" {
#endif

    // Function prototypes from your cqt.c file
    // Computes the CQT of audio. Returns freq_bins x time_frames complex values, to be freed with free_cqt_result.
    CQTResult* compute_cqt(const float* audio, size_t audio_length, float sample_rate,
                          int hop_length, float fmin, int n_bins, int bins_per_octave);

    // Frees the result struct and its data array.
    void free_cqt_result(CQTResult* result);

    // Frees the window cache and kernel cache that compute_cqt builds on first use.
    void cleanup_cqt_caches(void);

#ifdef __cplusplus
}
#endif

#endif // CQT_PROJECT_CQT_H