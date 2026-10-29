#ifndef SOUND_ACQUISITION_H
#define SOUND_ACQUISITION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Records mono 16-bit audio from an ALSA device until Ctrl+C. Returns normalised floats and sets the sample count.
float* record_audio_to_float(const char* device_name, unsigned int sample_rate, size_t* out_num_samples);

// Declared here only. There is no definition in sound_acquisition.c.
int sound_acquisition_main(int argc, char *argv[]);

#ifdef __cplusplus
}
#endif

#endif // SOUND_ACQUISITION_H