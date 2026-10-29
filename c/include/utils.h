#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Reads a raw float32 file into a new array and sets the element count. Exits on any error.
float* load_binary_file(const char* path, long* num_elements);
// Writes num_elements floats to a raw binary file. Exits on any error.
void write_binary_file(const char* path, const float* data, long num_elements);

#ifdef __cplusplus
}
#endif

#endif