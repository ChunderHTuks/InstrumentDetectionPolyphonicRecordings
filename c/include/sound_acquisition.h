#ifndef SOUND_ACQUISITION_H
#define SOUND_ACQUISITION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

float* record_audio_to_float(const char* device_name, unsigned int sample_rate, size_t* out_num_samples);

int sound_acquisition_main(int argc, char *argv[]);

#ifdef __cplusplus
}
#endif

#endif // SOUND_ACQUISITION_H