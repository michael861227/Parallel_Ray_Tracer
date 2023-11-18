#include <fstream>
#include "camera_t.h"
#include "sphere_t.h"
#include "scene_t.h"

int main() {
    // camera
    vec3_t lookfrom(0.5f, 0.5f, 1.0f);
    vec3_t lookat(0.5f, 0.5f, 0.0f);
    vec3_t vup(0.0f, 1.0f, 0.0f);
    float vfov = 55.0f;
    float aspect_ratio = 1.0f;
    camera_t camera(lookfrom, lookat, vup, vfov, aspect_ratio);

    // scene
    scene_t scene;
    scene.spheres.emplace_back(vec3_t(0.5f, 0.5f, -0.5f), 0.5f);
    scene.add_rectangle(vec3_t(0.0f, 0.0f, 0.0f),
                        vec3_t(0.0f, 1.0f, 0.0f),
                        vec3_t(0.0f, 1.0f, -1.0f),
                        vec3_t(0.0f, 0.0f, -1.0f));
    scene.add_rectangle(vec3_t(0.0f, 0.0f, -1.0f),
                        vec3_t(0.0f, 1.0f, -1.0f),
                        vec3_t(1.0f, 1.0f, -1.0f),
                        vec3_t(1.0f, 0.0f, -1.0f));
    scene.add_rectangle(vec3_t(1.0f, 0.0f, 0.0f),
                        vec3_t(1.0f, 1.0f, 0.0f),
                        vec3_t(1.0f, 1.0f, -1.0f),
                        vec3_t(1.0f, 0.0f, -1.0f));
    scene.add_rectangle(vec3_t(0.0f, 1.0f, 0.0f),
                        vec3_t(0.0f, 1.0f, -1.0f),
                        vec3_t(1.0f, 1.0f, -1.0f),
                        vec3_t(1.0f, 1.0f, 0.0f));
    scene.add_rectangle(vec3_t(0.0f, 0.0f, 0.0f),
                        vec3_t(0.0f, 0.0f, -1.0f),
                        vec3_t(1.0f, 0.0f, -1.0f),
                        vec3_t(1.0f, 0.0f, 0.0f));

    // output
    int image_width = 600;
    int image_height = 600;
    std::ofstream image_fs("image.ppm");
    image_fs << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            float s = float(i) / float(image_height - 1);
            float t = float(j) / float(image_height - 1);
            ray_t ray = camera.get_ray(s, t);
            bool hit = false;
            record_t record{};
            for (auto &sphere : scene.spheres)
                if (sphere.hit(ray, record))
                    hit = true;
            for (auto &trig : scene.trigs)
                if (trig.hit(ray, record))
                    hit = true;
            if (hit) {
                vec3_t color = (record.unit_n + vec3_t(1.0f, 1.0f, 1.0f)) / 2;
                color.write_color(image_fs);
            } else {
                vec3_t(0.5, 0.5, 0.5).write_color(image_fs);
            }
        }
    }

    return 0;
}
