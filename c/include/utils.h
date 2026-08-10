#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

float* load_binary_file(const char* path, long* num_elements);
void write_binary_file(const char* path, const float* data, long num_elements);

#ifdef __cplusplus
}
#endif

#endif