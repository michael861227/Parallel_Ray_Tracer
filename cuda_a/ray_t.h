#ifndef PARALLEL_RAY_TRACER_RAY_T_H
#define PARALLEL_RAY_TRACER_RAY_T_H

#include "vec3_t.h"

struct ray_t {
    __device__ vec3_t at(float t) const {
        return origin + t * direction;
    }

    vec3_t origin;
    vec3_t direction;
    float t_min;
    float t_max;
};

#endif //PARALLEL_RAY_TRACER_RAY_T_H