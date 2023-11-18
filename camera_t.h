#ifndef PARALLEL_RAY_TRACER_CAMERA_T_H
#define PARALLEL_RAY_TRACER_CAMERA_T_H

#include <cmath>
#include "ray_t.h"

class camera_t {
public:
    camera_t(
        vec3_t lookfrom,
        vec3_t lookat,
        vec3_t vup,
        float vfov_deg,
        float aspect_ratio
    ) {
        float vfov_rad = degree_to_radian(vfov_deg);
        float viewpoint_height = 2.0f * std::tan(vfov_rad / 2);
        float viewpoint_width = viewpoint_height * aspect_ratio;

        vec3_t w = (lookfrom - lookat).unit_vector();
        v = -(vup - dot(vup, w) * w).unit_vector();
        u = cross(v, w);

        origin = lookfrom;
        horizontal = viewpoint_width * u;
        vertical = viewpoint_height * v;
        upper_left_corner = origin - vertical / 2 - horizontal / 2 - w;
    }

    ray_t get_ray(float s, float t) {
        return {
            origin,
            upper_left_corner + s * vertical + t * horizontal - origin
        };
    }

private:
    vec3_t u;
    vec3_t v;
    vec3_t origin;
    vec3_t vertical;
    vec3_t horizontal;
    vec3_t upper_left_corner;
};

#endif //PARALLEL_RAY_TRACER_CAMERA_T_H
