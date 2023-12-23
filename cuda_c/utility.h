#ifndef PARALLEL_RAY_TRACER_UTILITY_H
#define PARALLEL_RAY_TRACER_UTILITY_H

#include <curand_kernel.h>
#include "../common/constants.h"

const int BLOCK_SIZE = 64;
const int NUM_WORKING_PATHS = 512;
const int IMAGE_WIDTH = 600;
const int IMAGE_HEIGHT = 600;
const int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
const int RAND_SEED = 0;

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
