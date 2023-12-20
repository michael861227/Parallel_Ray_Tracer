#include <fstream>
#include "camera_t.h"
#include "scene_t.h"
#include "render.h"

__global__ void render_kernel(camera_t* d_camera, scene_t* d_scene, vec3_t* d_framebuffer,
                              unsigned int image_width, unsigned int image_height) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int thread_id = y * gridDim.x * blockDim.x + x;
    if (x >= image_width || y >= image_height)
        return;

    curandState rand_state;
    curand_init(0, thread_id, 0, &rand_state);
    float s = float(x) / float(image_width - 1);
    float t = 1.0f - float(y) / float(image_height - 1);
    ray_t camera_ray = d_camera->get_ray(s, t);
    vec3_t color = vec3_t::make_zeros();
    for (int k = 1; k <= SAMPLES_PER_PIXEL; k++)
        color = color + get_color(*d_scene, camera_ray, rand_state);
    d_framebuffer[y * image_width + x] = color / SAMPLES_PER_PIXEL;
}

int main() {
    // camera
    vec3_t lookfrom(0.5f, 0.5f, 1.0f);
    vec3_t lookat(0.5f, 0.5f, 0.0f);
    vec3_t vup(0.0f, 1.0f, 0.0f);
    float vfov = 55.0f;
    float aspect_ratio = 1.0f;
    camera_t camera(lookfrom, lookat, vup, vfov, aspect_ratio);
    camera_t* d_camera;
    CHECK_CUDA(cudaMalloc(&d_camera, sizeof(camera_t)));
    CHECK_CUDA(cudaMemcpy(d_camera, &camera, sizeof(camera_t), cudaMemcpyHostToDevice));

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
    scene_t* d_scene;
    CHECK_CUDA(cudaMalloc(&d_scene, sizeof(scene_t)));
    CHECK_CUDA(cudaMemcpy(d_scene, &scene, sizeof(scene_t), cudaMemcpyHostToDevice));

    // render
    unsigned int image_width = 600;
    unsigned int image_height = 600;

    vec3_t* d_framebuffer;
    CHECK_CUDA(cudaMalloc(&d_framebuffer, image_height * image_width * sizeof(vec3_t)));
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size((image_width + block_size.x - 1) / block_size.x, (image_height + block_size.y - 1) / block_size.y);
    render_kernel<<<grid_size, block_size>>>(d_camera, d_scene, d_framebuffer, image_width, image_height);

    // write framebuffer to file
    vec3_t framebuffer[image_height * image_width];
    CHECK_CUDA(cudaMemcpy(framebuffer, d_framebuffer, image_height * image_width * sizeof(vec3_t), cudaMemcpyDeviceToHost));
    std::ofstream image_fs("image.ppm");
    image_fs << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int i = 0; i < image_height; i++)
        for (int j = 0; j < image_width; j++)
            framebuffer[i * image_width + j].write_color(image_fs);

    return 0;
}
