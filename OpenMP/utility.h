#ifndef PARALLEL_RAY_TRACER_UTILITY_H
#define PARALLEL_RAY_TRACER_UTILITY_H

#include "../common/constants.h"

float random_float() {
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    static std::mt19937 generator;
    return distribution(generator);
}

float degree_to_radian(float degree) {
    return degree * PI / 180.0f;
}

#endif //PARALLEL_RAY_TRACER_UTILITY_H
