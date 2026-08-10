#include "utils.h"
#include <string.h>

float* load_binary_file(const char* path, long* num_elements) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", path);
        exit(EXIT_FAILURE);
    }
    
    if (fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "Failed to seek to end of file: %s\n", path);
        fclose(f);
        exit(EXIT_FAILURE);
    }
    
    long file_size = ftell(f);
    if (file_size < 0) {
        fprintf(stderr, "Failed to get file size: %s\n", path);
        fclose(f);
        exit(EXIT_FAILURE);
    }
    
    if (fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "Failed to seek to beginning of file: %s\n", path);
        fclose(f);
        exit(EXIT_FAILURE);
    }
    
    *num_elements = file_size / sizeof(float);
    if (*num_elements == 0) {
        fprintf(stderr, "Empty or invalid file: %s\n", path);
        fclose(f);
        exit(EXIT_FAILURE);
    }
    
    float* data = (float*)malloc(file_size);
    if (!data) {
        fprintf(stderr, "Failed to allocate %ld bytes for file: %s\n", file_size, path);
        fclose(f);
        exit(EXIT_FAILURE);
    }
    
    size_t elements_read = fread(data, sizeof(float), *num_elements, f);
    if (elements_read != (size_t)*num_elements) {
        fprintf(stderr, "Failed to read all elements from file: %s (got %zu, expected %ld)\n", 
                path, elements_read, *num_elements);
        free(data);
        fclose(f);
        exit(EXIT_FAILURE);
    }
    
    fclose(f);
    return data;
}

void write_binary_file(const char* path, const float* data, long num_elements) {
    if (!data) {
        fprintf(stderr, "Null data pointer for file: %s\n", path);
        exit(EXIT_FAILURE);
    }
    
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Failed to open file for writing: %s\n", path);
        exit(EXIT_FAILURE);
    }
    
    size_t elements_written = fwrite(data, sizeof(float), num_elements, f);
    if (elements_written != (size_t)num_elements) {
        fprintf(stderr, "Failed to write all elements to file: %s (wrote %zu, expected %ld)\n", 
                path, elements_written, num_elements);
        fclose(f);
        exit(EXIT_FAILURE);
    }
    
    if (fclose(f) != 0) {
        fprintf(stderr, "Failed to close file: %s\n", path);
        exit(EXIT_FAILURE);
    }
}