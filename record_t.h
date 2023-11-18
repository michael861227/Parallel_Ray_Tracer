#ifndef PARALLEL_RAY_TRACER_RECORD_T_H
#define PARALLEL_RAY_TRACER_RECORD_T_H

struct record_t {
    vec3_t p;         // hit point
    bool front_face;  // hit front face?
    vec3_t n;         // unit normal vector of hit point
};

#endif //PARALLEL_RAY_TRACER_RECORD_T_H
