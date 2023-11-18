#ifndef PARALLEL_RAY_TRACER_UTILITY_H
#define PARALLEL_RAY_TRACER_UTILITY_H

const float PI = 3.14159265f;

template<int min, int max>
float random_float() {
    static std::uniform_real_distribution<float> distribution(min, max);
    static std::mt19937 generator;
    return distribution(generator);
}

float degrees_to_radians(float degrees) {
    return degrees * PI / 180.0f;
}

#endif //PARALLEL_RAY_TRACER_UTILITY_H
