#ifndef PARALLEL_RAY_TRACER_SCENE_T_H
#define PARALLEL_RAY_TRACER_SCENE_T_H

#include "sphere_t.h"
#include "trig_t.h"

struct point_light_t {
    point_light_t(const vec3_t &position, const vec3_t &intensity)
        : position(position), intensity(intensity) {}

    vec3_t position;
    vec3_t intensity;
};

struct scene_t {
    std::vector<sphere_t> spheres;
    std::vector<trig_t> trigs;
    std::vector<point_light_t> point_lights;

    void add_rectangle(const vec3_t &p0, const vec3_t &p1, const vec3_t &p2, const vec3_t &p3, const vec3_t &albedo) {
        trigs.emplace_back(p0, p1, p2, albedo);
        trigs.emplace_back(p2, p3, p0, albedo);
    }
};

#endif //PARALLEL_RAY_TRACER_SCENE_T_H
