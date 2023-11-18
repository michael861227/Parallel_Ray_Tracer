#ifndef PARALLEL_RAY_TRACER_RAY_T_H
#define PARALLEL_RAY_TRACER_RAY_T_H

class ray_t {
public:
    ray_t(const vec3_t &origin, const vec3_t &direction) : origin(origin), direction(direction) {}

    vec3_t at(float t) const {
        return origin + t * direction;
    }

    vec3_t origin;
    vec3_t direction;
};

#endif //PARALLEL_RAY_TRACER_RAY_T_H