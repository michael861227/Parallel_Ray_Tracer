#include <fstream>
#include "camera_t.h"
#include "scene_t.h"
#include "render.h"

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
    scene.spheres.emplace_back(vec3_t(0.5f, 0.5f, -0.5f), 0.2f);
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
    scene.point_lights.emplace_back(vec3_t(0.5f, 0.9f, 0.8f), vec3_t(1.0f, 1.0f, 1.0f));

    // output
    int image_width = 600;
    int image_height = 600;
    std::ofstream image_fs("image.ppm");
    image_fs << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            float s = float(j) / float(image_width - 1);
            float t = 1.0f - float(i) / float(image_height - 1);
            ray_t ray = camera.get_ray(s, t);
            get_color(scene, ray).write_color(image_fs);
        }
    }

    return 0;
}
