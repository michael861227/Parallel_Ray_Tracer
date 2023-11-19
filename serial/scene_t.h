#ifndef PARALLEL_RAY_TRACER_SCENE_T_H
#define PARALLEL_RAY_TRACER_SCENE_T_H

#include "sphere_t.h"
#include "trig_t.h"
#include "point_light_t.h"

struct scene_t {
    std::vector<sphere_t> spheres;
    std::vector<trig_t> trigs;
    point_light_t point_light;

    void add_rectangle(const vec3_t &p0, const vec3_t &p1, const vec3_t &p2, const vec3_t &p3, const vec3_t &albedo) {
        trigs.emplace_back(p0, p1, p2, albedo);
        trigs.emplace_back(p2, p3, p0, albedo);
    }
};

#endif //PARALLEL_RAY_TRACER_SCENE_T_H
