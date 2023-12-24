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
    camera_t* d_camera_ptr;
    CHECK_CUDA(cudaMalloc(&d_camera_ptr, sizeof(camera_t)));
    CHECK_CUDA(cudaMemcpy(d_camera_ptr, &camera, sizeof(camera_t), cudaMemcpyHostToDevice));

    // color
    vec3_t red(0.65f, 0.05f, 0.05f);
    vec3_t green(0.12f, 0.45f, 0.15f);
    vec3_t white(0.73f, 0.73f, 0.73f);
    vec3_t brown(0.62f, 0.57f, 0.54f);

    // scene
    std::vector<sphere_t> spheres;
    std::vector<trig_t> trigs;
    auto add_rectangle = [&](const vec3_t &p0, const vec3_t &p1, const vec3_t &p2,
                             const vec3_t &p3, const vec3_t &albedo) {
        trigs.emplace_back(p0, p1, p2, albedo);
        trigs.emplace_back(p2, p3, p0, albedo);
    };
    spheres.emplace_back(vec3_t(0.5f, 0.2f, -0.25f), 0.2f, brown);
    add_rectangle(vec3_t(0.0f, 0.0f, 0.0f),
                  vec3_t(0.0f, 1.0f, 0.0f),
                  vec3_t(0.0f, 1.0f, -1.0f),
                  vec3_t(0.0f, 0.0f, -1.0f),
                  red);
    add_rectangle(vec3_t(0.0f, 0.0f, -1.0f),
                  vec3_t(0.0f, 1.0f, -1.0f),
                  vec3_t(1.0f, 1.0f, -1.0f),
                  vec3_t(1.0f, 0.0f, -1.0f),
                  white);
    add_rectangle(vec3_t(1.0f, 0.0f, 0.0f),
                  vec3_t(1.0f, 1.0f, 0.0f),
                  vec3_t(1.0f, 1.0f, -1.0f),
                  vec3_t(1.0f, 0.0f, -1.0f),
                  green);
    add_rectangle(vec3_t(0.0f, 1.0f, 0.0f),
                  vec3_t(0.0f, 1.0f, -1.0f),
                  vec3_t(1.0f, 1.0f, -1.0f),
                  vec3_t(1.0f, 1.0f, 0.0f),
                  white);
    add_rectangle(vec3_t(0.0f, 0.0f, 0.0f),
                  vec3_t(0.0f, 0.0f, -1.0f),
                  vec3_t(1.0f, 0.0f, -1.0f),
                  vec3_t(1.0f, 0.0f, 0.0f),
                  white);
    scene_t scene{};
    scene.num_spheres = (int)spheres.size();
    CHECK_CUDA(cudaMalloc(&scene.spheres, scene.num_spheres * sizeof(sphere_t)));
    CHECK_CUDA(cudaMemcpy(scene.spheres, spheres.data(), scene.num_spheres * sizeof(sphere_t), cudaMemcpyHostToDevice));
    scene.num_trigs = (int)trigs.size();
    CHECK_CUDA(cudaMalloc(&scene.trigs, scene.num_trigs * sizeof(trig_t)));
    CHECK_CUDA(cudaMemcpy(scene.trigs, trigs.data(), scene.num_trigs * sizeof(trig_t), cudaMemcpyHostToDevice));
    scene.point_light = {vec3_t(0.95f, 0.95f, 0.3f), vec3_t(0.9f, 0.9f, 0.9f)};
    scene_t* d_scene_ptr;
    CHECK_CUDA(cudaMalloc(&d_scene_ptr, sizeof(scene_t)));
    CHECK_CUDA(cudaMemcpy(d_scene_ptr, &scene, sizeof(scene_t), cudaMemcpyHostToDevice));

    // render
    float* d_framebuffer_x_ptr;
    float* d_framebuffer_y_ptr;
    float* d_framebuffer_z_ptr;
    CHECK_CUDA(cudaMalloc(&d_framebuffer_x_ptr, NUM_PIXELS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_framebuffer_y_ptr, NUM_PIXELS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_framebuffer_z_ptr, NUM_PIXELS * sizeof(float)));
    render(d_camera_ptr, d_scene_ptr, d_framebuffer_x_ptr, d_framebuffer_y_ptr, d_framebuffer_z_ptr);

    // write framebuffer to file
    float framebuffer_x[NUM_PIXELS];
    float framebuffer_y[NUM_PIXELS];
    float framebuffer_z[NUM_PIXELS];
    CHECK_CUDA(cudaMemcpy(framebuffer_x, d_framebuffer_x_ptr, NUM_PIXELS * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(framebuffer_y, d_framebuffer_y_ptr, NUM_PIXELS * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(framebuffer_z, d_framebuffer_z_ptr, NUM_PIXELS * sizeof(float), cudaMemcpyDeviceToHost));
    std::ofstream image_fs("image.ppm");
    image_fs << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << "\n255\n";
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            vec3_t color = {
                framebuffer_x[i * IMAGE_WIDTH + j] / SAMPLES_PER_PIXEL,
                framebuffer_y[i * IMAGE_WIDTH + j] / SAMPLES_PER_PIXEL,
                framebuffer_z[i * IMAGE_WIDTH + j] / SAMPLES_PER_PIXEL
            };
            color.write_color(image_fs);
        }
    }

    return 0;
}
