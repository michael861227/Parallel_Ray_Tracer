#ifndef PARALLEL_RAY_TRACER_SPHERE_T_H
#define PARALLEL_RAY_TRACER_SPHERE_T_H

#include "vec3_t.h"
#include "record_t.h"

class sphere_t {
public:
    sphere_t(const vec3_t &center, float radius, const vec3_t &albedo)
        : center(center), radius(radius), albedo(albedo) {}

    __device__ bool hit(ray_t &ray, record_t &record) const {
        vec3_t amc = ray.origin - center;
        float a = dot(ray.direction, ray.direction);
        float b = 2.0f * dot(amc, ray.direction);
        float c = dot(amc, amc) - radius * radius;
        float delta = b * b - 4 * a * c;
        if (delta < 0)
            return false;

        float inv = -0.5f / a;
        float sqrt_delta = std::sqrt(delta);

        float root = (b + sqrt_delta) * inv;
        if (!(ray.t_min <= root && root <= ray.t_max)) {
            root = (b - sqrt_delta) * inv;
            if (!(ray.t_min <= root && root <= ray.t_max))
                return false;
        }

        ray.t_max = root;
        record.hit_point = ray.at(root);
        vec3_t outward_normal = (record.hit_point - center) / radius;
        bool front_face = dot(ray.direction, outward_normal) < 0;
        record.unit_n = front_face ? outward_normal : -outward_normal;
        record.albedo = albedo;

        return true;
    }

private:
    vec3_t center;
    float radius;
    vec3_t albedo;
};

#endif //PARALLEL_RAY_TRACER_SPHERE_T_H
