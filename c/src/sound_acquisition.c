#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <alsa/asoundlib.h>
#include "sound_acquisition.h"

// Capture settings. Mono 16-bit signed little-endian samples, read in blocks of 1024 frames.
#define CHANNELS 1
#define SAMPLE_FORMAT SND_PCM_FORMAT_S16_LE
#define BUFFER_FRAMES 1024

// Cleared by Ctrl+C or SIGTERM to end the capture loop.
static volatile int running = 1;

// Stops the capture loop. Only sets the flag, because little else is safe inside a signal handler.
static void signal_handler(int sig) {
    (void)sig;
    running = 0;
}

// Records mono audio from an ALSA capture device until Ctrl+C or SIGTERM.
// Opens the device and sets interleaved 16-bit access, the requested sample rate and one channel.
// Reads blocks of frames into a small buffer and appends them to a growing int16 buffer.
// Doubles the growing buffer when it is full, because the recording length is not known in advance.
// Recovers from an overrun (EPIPE) by preparing the device again.
// Converts the int16 samples to float by dividing by 32768, giving values in [-1, 1).
// Returns the float array and sets out_num_samples. Returns NULL on failure.
float* record_audio_to_float(const char* device_name, unsigned int sample_rate, size_t* out_num_samples) {
    snd_pcm_t *capture_handle;
    snd_pcm_hw_params_t *hw_params;
    snd_pcm_uframes_t frames = BUFFER_FRAMES;
    int16_t *buffer;
    int err;

    *out_num_samples = 0;

    if ((err = snd_pcm_open(&capture_handle, device_name, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        fprintf(stderr, "Cannot open audio device %s (%s)\n", device_name, snd_strerror(err));
        return NULL;
    }

    snd_pcm_hw_params_malloc(&hw_params);
    snd_pcm_hw_params_any(capture_handle, hw_params);
    snd_pcm_hw_params_set_access(capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(capture_handle, hw_params, SAMPLE_FORMAT);
    unsigned int rate = sample_rate;
    snd_pcm_hw_params_set_rate_near(capture_handle, hw_params, &rate, 0);
    if (rate != sample_rate) {
        fprintf(stderr, "Warning: Requested rate %u, got %u\n", sample_rate, rate);
    }
    snd_pcm_hw_params_set_channels(capture_handle, hw_params, CHANNELS);
    snd_pcm_hw_params_set_buffer_size_near(capture_handle, hw_params, &frames);
    snd_pcm_hw_params(capture_handle, hw_params);
    snd_pcm_hw_params_free(hw_params);
    snd_pcm_prepare(capture_handle);

    buffer = malloc(frames * CHANNELS * sizeof(int16_t));
    if (!buffer) {
        fprintf(stderr, "Cannot allocate buffer\n");
        snd_pcm_close(capture_handle);
        return NULL;
    }

    size_t capacity = frames * 100;
    int16_t* in_memory_buffer = malloc(capacity * sizeof(int16_t));
    if (!in_memory_buffer) {
        fprintf(stderr, "Cannot allocate in-memory buffer\n");
        free(buffer);
        snd_pcm_close(capture_handle);
        return NULL;
    }
    size_t total_frames = 0;

    running = 1;
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("Recording from device: %s... Press Ctrl+C to stop.\n", device_name);

    while (running) {
        err = snd_pcm_readi(capture_handle, buffer, frames);
        
        if (err > 0) {
            if (total_frames + err > capacity) {
                capacity *= 2;
                int16_t* new_buffer = realloc(in_memory_buffer, capacity * sizeof(int16_t));
                if (!new_buffer) {
                    fprintf(stderr, "Failed to reallocate in-memory buffer\n");
                    free(in_memory_buffer);
                    free(buffer);
                    snd_pcm_close(capture_handle);
                    return NULL;
                }
                in_memory_buffer = new_buffer;
            }
            memcpy(in_memory_buffer + total_frames, buffer, err * sizeof(int16_t));
            total_frames += err;
        } else if (err == -EPIPE) {
            fprintf(stderr, "Buffer overrun occurred\n");
            snd_pcm_prepare(capture_handle);
        } else if (err < 0) {
            fprintf(stderr, "Error from read: %s\n", snd_strerror(err));
            running = 0;
        }
    }

    signal(SIGINT, SIG_DFL);
    signal(SIGTERM, SIG_DFL);

    printf("\nRecording stopped. Total samples: %zu (%.2f seconds)\n", 
           total_frames, (float)total_frames / sample_rate);

    free(buffer);
    snd_pcm_close(capture_handle);

    float* float_buffer = malloc(total_frames * sizeof(float));
    if (!float_buffer) {
        fprintf(stderr, "Failed to allocate final float buffer\n");
        free(in_memory_buffer);
        return NULL;
    }

    for (size_t i = 0; i < total_frames; i++) {
        float_buffer[i] = (float)in_memory_buffer[i] / 32768.0f;
    }

    free(in_memory_buffer);
    *out_num_samples = total_frames;
    
    printf("Audio converted to 32-bit float and normalized.\n");
    
    return float_buffer;
}