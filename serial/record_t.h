#ifndef PARALLEL_RAY_TRACER_RECORD_T_H
#define PARALLEL_RAY_TRACER_RECORD_T_H

struct record_t {
    vec3_t hit_point;
    vec3_t unit_n;  // unit normal vector of hit point
    vec3_t albedo;  // albedo of the object being hit
};

#endif //PARALLEL_RAY_TRACER_RECORD_T_H
