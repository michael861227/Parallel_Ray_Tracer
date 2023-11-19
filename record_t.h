#ifndef PARALLEL_RAY_TRACER_RECORD_T_H
#define PARALLEL_RAY_TRACER_RECORD_T_H

struct record_t {
    vec3_t hit_point;
    vec3_t unit_n;    // unit normal vector of hit point
    const vec3_t* albedo;
};

#endif //PARALLEL_RAY_TRACER_RECORD_T_H
