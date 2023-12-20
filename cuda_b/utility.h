#ifndef PARALLEL_RAY_TRACER_UTILITY_H
#define PARALLEL_RAY_TRACER_UTILITY_H

#include <curand_kernel.h>

const float PI = 3.14159265f;
const float EPS = 0.01f;
const int MAX_PATH = 2;
const int SAMPLES_PER_PIXEL = 256;
const int BLOCK_SIZE_X = 1;
const int BLOCK_SIZE_Y = 1;
const int BLOCK_SIZE_Z = 128;

#define CHECK_CUDA(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

__device__ float random_float(curandState &rand_state) {
    return curand_uniform(&rand_state);
}

float degree_to_radian(float degree) {
    return degree * PI / 180.0f;
}

#endif //PARALLEL_RAY_TRACER_UTILITY_H
