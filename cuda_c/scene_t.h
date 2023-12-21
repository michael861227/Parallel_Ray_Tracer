#ifndef PARALLEL_RAY_TRACER_SCENE_T_H
#define PARALLEL_RAY_TRACER_SCENE_T_H

#include "sphere_t.h"
#include "trig_t.h"
#include "point_light_t.h"

struct scene_t {
    int num_spheres;
    sphere_t* spheres;
    int num_trigs;
    trig_t* trigs;
    point_light_t point_light;
};

#endif //PARALLEL_RAY_TRACER_SCENE_T_H
