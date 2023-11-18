#ifndef PARALLEL_RAY_TRACER_SCENE_T_H
#define PARALLEL_RAY_TRACER_SCENE_T_H

#include "sphere_t.h"
#include "trig_t.h"

struct point_light_t {
    vec3_t p;
    vec3_t color;
};

struct scene_t {
    std::vector<sphere_t> spheres;
    std::vector<trig_t> trigs;
    std::vector<point_light_t> lights;

    void add_rectangle(const vec3_t &p0, const vec3_t &p1, const vec3_t &p2, const vec3_t &p3) {
        trigs.emplace_back(p0, p1, p2);
        trigs.emplace_back(p2, p3, p0);
    }
};

#endif //PARALLEL_RAY_TRACER_SCENE_T_H
