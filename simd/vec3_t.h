#ifndef PARALLEL_RAY_TRACER_VEC3_T_H
#define PARALLEL_RAY_TRACER_VEC3_T_H
#define PADDING 0.0f

#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>
#include "utility.h"
#include <immintrin.h>



class vec3_t {
public:
    vec3_t() = default;

    vec3_t(float x, float y, float z) {
        this->v = _mm_set_ps(PADDING, z, y, x);
    }

    vec3_t(__m128 vector) {
        this->v = vector;
    }

    vec3_t operator-() const {
        __m128 negOne = _mm_set_ps1(-1.0f);
        return { _mm_mul_ps(this->v, negOne) };
    }

    float operator[](int idx) const {
        idx = 3 - idx;
        float element = _mm_cvtss_f32(_mm_shuffle_ps(v, v, _MM_SHUFFLE(idx, idx, idx, idx)));
        return element;
    }

    float length_squared() const {

        __m128 squared_v = _mm_mul_ps(v, v);
        return get_vector_sum(squared_v);
    }

    float length() const {
        return std::sqrt(length_squared());
    }

    vec3_t unit_vector() const {
        // return { v[0] / len, v[1] / len, v[2] / len };
        float len = length();
        __m128 len_v = _mm_set1_ps(len);
        return { _mm_div_ps(v, len_v) };
    }

    void write_color(std::ostream& out) {
        // color: [0, 1]
        // float r = std::sqrt(std::clamp(v[0], 0.0f, 1.0f));
        // float g = std::sqrt(std::clamp(v[1], 0.0f, 1.0f));
        // float b = std::sqrt(std::clamp(v[2], 0.0f, 1.0f));
        // int rr = (int)std::round(255.0f * r);
        // int gg = (int)std::round(255.0f * g);
        // int bb = (int)std::round(255.0f * b);
        __m128 rgb_v = _mm_min_ps(_mm_max_ps(v, _mm_setzero_ps()), _mm_set_ps1(1.0f));
        rgb_v = _mm_sqrt_ps(rgb_v);
        rgb_v = _mm_mul_ps(rgb_v, _mm_set1_ps(255.0f));
        float rgb[4];
        vector2array(rgb_v, rgb);
        int rr = round(rgb[0]);
        int gg = round(rgb[1]);
        int bb = round(rgb[2]);
        // std::cout << rr << ' ' << gg << ' ' << bb << '\n';
        assert(0 <= rr && rr <= 255);
        assert(0 <= gg && gg <= 255);
        assert(0 <= bb && bb <= 255);
        out << rr << ' ' << gg << ' ' << bb << '\n';
    }

    static vec3_t uniform_sample_sphere() {
        float z = 1 - 2 * random_float();
        float r = sqrtf(1 - z * z);
        float phi = 2 * PI * random_float();
        float x = std::cos(phi);
        float y = std::sin(phi);
        return { _mm_set_ps(PADDING, r * x, r * y, z) };
        // return { r * x, r * y, z };
    }

    static vec3_t make_zeros() {
        // return { 0.0f, 0.0f, 0.0f };
        return { _mm_setzero_ps() };
    }

    static vec3_t make_ones() {
        // return { 1.0f, 1.0f, 1.0f };
        return { _mm_set1_ps(1.0f) };
    }

    __m128 get_vector() const {
        return this->v;
    }

    void vector2array(__m128 v, float* arr) const {
        _mm_storeu_ps(arr, v);
    }

    float get_vector_sum(__m128 v) const {
        __m128 sumVector = _mm_hadd_ps(v, v);
        sumVector = _mm_hadd_ps(sumVector, sumVector);
        float result;
        _mm_store_ss(&result, sumVector);
        return result;
    }

private:
    __m128 v;
};


vec3_t operator+(const vec3_t& v1, const vec3_t& v2) {
    return { _mm_add_ps(v1.get_vector(), v2.get_vector()) };
    // return { v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2] };
}

vec3_t operator-(const vec3_t& v1, const vec3_t& v2) {
    return v1 + (-v2);
}

vec3_t operator*(const vec3_t& v, float t) {
    // return { v[0] * t, v[1] * t, v[2] * t };
    return { _mm_mul_ps(v.get_vector(), _mm_set1_ps(t)) };
}

vec3_t operator*(float t, const vec3_t& v) {
    // return v * t;
    return { _mm_mul_ps(_mm_set1_ps(t), v.get_vector()) };
}

vec3_t operator*(const vec3_t& v1, const vec3_t& v2) {
    // return { v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2] };
    return { _mm_mul_ps(v1.get_vector(), v2.get_vector()) };
}

vec3_t operator/(const vec3_t& v, float t) {
    // return { v[0] / t, v[1] / t, v[2] / t };
    return { _mm_div_ps(v.get_vector(), _mm_set1_ps(t)) };
}

float dot(const vec3_t& u, const vec3_t& v) {
    // return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    vec3_t mul = u * v;
    return mul.get_vector_sum(mul.get_vector());
}

vec3_t cross(const vec3_t& u, const vec3_t& v) {
    float arr_u[4], arr_v[4];
    u.vector2array(u.get_vector(), arr_u);
    v.vector2array(v.get_vector(), arr_v);
    return { arr_u[1] * arr_v[2] - arr_u[2] * arr_v[1],
            arr_u[2] * arr_v[0] - arr_u[0] * arr_v[2],
            arr_u[0] * arr_v[1] - arr_u[1] * arr_v[0] };
}

#endif //PARALLEL_RAY_TRACER_VEC3_T_H
