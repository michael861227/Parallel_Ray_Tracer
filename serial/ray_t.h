#ifndef PARALLEL_RAY_TRACER_RAY_T_H
#define PARALLEL_RAY_TRACER_RAY_T_H

#include "vec3_t.h"

class ray_t {
public:
    ray_t(const vec3_t &origin, const vec3_t &direction, float t_min, float t_max) :
        origin(origin), direction(direction), t_min(t_min), t_max(t_max) {}

    vec3_t at(float t) const {
        return origin + t * direction;
    }

    vec3_t origin;
    vec3_t direction;
    float t_min;
    float t_max;
};

#endif //PARALLEL_RAY_TRACER_RAY_T_H