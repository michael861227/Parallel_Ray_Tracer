#ifndef PARALLEL_RAY_TRACER_UTILITY_H
#define PARALLEL_RAY_TRACER_UTILITY_H

const float PI = 3.14159265f;
const float EPS = 0.01f;
const int PATH_MAX = 3;
const int SAMPLES_PER_PIXEL = 10;

template<int min, int max>
float random_float() {
    static std::uniform_real_distribution<float> distribution(min, max);
    static std::mt19937 generator;
    return distribution(generator);
}

float degree_to_radian(float degree) {
    return degree * PI / 180.0f;
}

#endif //PARALLEL_RAY_TRACER_UTILITY_H
