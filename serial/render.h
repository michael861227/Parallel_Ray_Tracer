#ifndef PARALLEL_RAY_TRACER_RENDER_H
#define PARALLEL_RAY_TRACER_RENDER_H

bool occluded(const scene_t &scene, ray_t &ray) {
    record_t record{};
    for (const auto &sphere : scene.spheres)
        if (sphere.hit(ray, record))
            return true;
    for (const auto &trig : scene.trigs)
        if (trig.hit(ray, record))
            return true;
    return false;
}

vec3_t get_color(const scene_t &scene, ray_t &ray) {
    vec3_t color = vec3_t::make_zeros();
    vec3_t multiplier = vec3_t::make_ones();

    for (int i = 1; i <= PATH_MAX; i++) {
        bool hit = false;
        record_t record{};
        for (const auto &sphere : scene.spheres)
            if (sphere.hit(ray, record))
                hit = true;
        for (const auto &trig : scene.trigs)
            if (trig.hit(ray, record))
                hit = true;

        if (hit) {
            vec3_t shadow_dir = scene.point_light.position - record.hit_point;

            multiplier = multiplier * (*record.albedo);
            if (dot(ray.direction, record.unit_n) * dot(shadow_dir, record.unit_n) < 0.f) {  // in same hemisphere
                ray_t shadow_ray = {record.hit_point, shadow_dir, EPS, 1.0f};
                if (!occluded(scene, shadow_ray)) {
                    float t2 = shadow_dir.length_squared();
                    float t = std::sqrt(t2);
                    color = color + multiplier * scene.point_light.intensity / t2 *
                                    dot(shadow_dir, record.unit_n) / t;
                }
            }

            ray = {record.hit_point,
                   record.unit_n + vec3_t::uniform_sample_sphere(),
                   EPS,
                   std::numeric_limits<float>::max()};
        } else {
            break;
        }
    }

    return color;
}

#endif //PARALLEL_RAY_TRACER_RENDER_H
