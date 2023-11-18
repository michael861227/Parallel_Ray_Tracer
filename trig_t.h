#ifndef PARALLEL_RAY_TRACER_TRIG_T_H
#define PARALLEL_RAY_TRACER_TRIG_T_H

#include "record_t.h"

class trig_t {
public:
    trig_t(const vec3_t &p0, const vec3_t &p1, const vec3_t &p2)
        : p0(p0), e1(p0 - p1), e2(p2 - p0), n(cross(e1, e2)) { }

    bool hit(ray_t &ray, record_t &rec) {
        vec3_t c = p0 - ray.origin;
        vec3_t r = cross(ray.direction, c);
        float inv_det = 1.f / dot(ray.direction, n);

        float u = inv_det * dot(e2, r);
        float v = inv_det * dot(e1, r);

        if (u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f) {
            float t = inv_det * dot(c, n);
            if (ray.t_min <= t && t <= ray.t_max) {
                ray.t_max = t;
                rec.p = ray.at(t);
                rec.front_face = dot(ray.direction, n) < 0;
                rec.n = rec.front_face ? n : -n;
                return true;
            }
        }

        return false;
    }
private:
    vec3_t p0, e1, e2, n;
};

#endif //PARALLEL_RAY_TRACER_TRIG_T_H
