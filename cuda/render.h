#ifndef PARALLEL_RAY_TRACER_RENDER_H
#define PARALLEL_RAY_TRACER_RENDER_H

__device__ bool occluded(const scene_t &scene, ray_t &ray) {
    record_t record{};
    for (int i = 0; i < scene.num_spheres; i++)
        if (scene.spheres[i].hit(ray, record))
            return true;
    for (int i = 0; i < scene.num_trigs; i++)
        if (scene.trigs[i].hit(ray, record))
            return true;
    return false;
}

__device__ vec3_t get_color(const scene_t &scene, ray_t ray, curandState &rand_state) {
    vec3_t color = vec3_t::make_zeros();
    vec3_t multiplier = vec3_t::make_ones();

    for (int i = 1; i <= MAX_PATH; i++) {
        bool hit = false;
        record_t record{};
        for (int j = 0; j < scene.num_spheres; j++)
            if (scene.spheres[j].hit(ray, record))
                hit = true;
        for (int j = 0; j < scene.num_trigs; j++)
            if (scene.trigs[j].hit(ray, record))
                hit = true;

        if (hit) {
            vec3_t shadow_dir = scene.point_light.position - record.hit_point;

            multiplier = multiplier * record.albedo;
            if (dot(ray.direction, record.unit_n) * dot(shadow_dir, record.unit_n) < 0.f) {  // in same hemisphere
                ray_t shadow_ray = {record.hit_point, shadow_dir, EPS, 1.0f};
                if (!occluded(scene, shadow_ray)) {
                    float t2 = shadow_dir.length_squared();
                    float t = std::sqrt(t2);
                    color = color + multiplier * scene.point_light.intensity / t2 *
                                    dot(shadow_dir, record.unit_n) / t;  // cos(theta)
                }
            }

            ray = {record.hit_point,
                   record.unit_n + vec3_t::uniform_sample_sphere(rand_state),
                   EPS,
                   FLT_MAX};
        } else {
            break;
        }
    }

    return color;
}

#endif //PARALLEL_RAY_TRACER_RENDER_H
