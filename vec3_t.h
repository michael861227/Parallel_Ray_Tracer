#ifndef PARALLEL_RAY_TRACER_VEC3_T_H
#define PARALLEL_RAY_TRACER_VEC3_T_H

#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>
#include "utility.h"

class vec3_t {
public:
    vec3_t() = default;

    vec3_t(float x, float y, float z) : v{x, y, z} {}

    vec3_t operator-() const {
        return {-v[0], -v[1], -v[2]};
    }

    float operator[](int x) const {
        return v[x];
    }

    float length_squared() {
        return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    }

    float length() {
        return std::sqrt(length_squared());
    }

    vec3_t unit_vector() {
        float len = length();
        return {v[0] / len, v[1] / len, v[2] / len};
    }

    void write_color(std::ostream &out) {
        // color: [0, 1]
        float r = std::sqrt(std::clamp(v[0], 0.0f, 1.0f));
        float g = std::sqrt(std::clamp(v[1], 0.0f, 1.0f));
        float b = std::sqrt(std::clamp(v[2], 0.0f, 1.0f));
        int rr = (int)std::round(255.0f * r);
        int gg = (int)std::round(255.0f * g);
        int bb = (int)std::round(255.0f * b);
        assert(0 <= rr && rr <= 255);
        assert(0 <= gg && gg <= 255);
        assert(0 <= bb && bb <= 255);
        out << rr << ' ' << gg << ' ' << bb << '\n';
    }

    template<int min, int max>
    static vec3_t random() {
        return {random_float<min, max>(), random_float<min, max>(), random_float<min, max>()};
    }

    static vec3_t random_in_unit_sphere() {
        while (true) {
            vec3_t p = vec3_t::random<-1, 1>();
            if (p.length_squared() <= 1)
                return p;
        }
    }

private:
    float v[3];
};

vec3_t operator+(const vec3_t &v1, const vec3_t &v2) {
    return {v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]};
}

vec3_t operator-(const vec3_t &v1, const vec3_t &v2) {
    return v1 + (-v2);
}

vec3_t operator*(const vec3_t &v, float t) {
    return {v[0] * t, v[1] * t, v[2] * t};
}

vec3_t operator*(float t, const vec3_t &v) {
    return v * t;
}

vec3_t operator*(const vec3_t &v1, const vec3_t &v2) {
    return {v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]};
}

vec3_t operator/(const vec3_t &v, float t) {
    return {v[0] / t, v[1] / t, v[2] / t};
}

float dot(const vec3_t &u, const vec3_t &v) {
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

vec3_t cross(const vec3_t &u, const vec3_t &v) {
    return {u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0]};
}

#endif //PARALLEL_RAY_TRACER_VEC3_T_H
