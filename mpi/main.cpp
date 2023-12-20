#include <fstream>
#include "camera_t.h"
#include "scene_t.h"
#include "render.h"
#include "mpi.h"

int main() {
    MPI_Init(nullptr, nullptr);

    // camera
    vec3_t lookfrom(0.5f, 0.5f, 1.0f);
    vec3_t lookat(0.5f, 0.5f, 0.0f);
    vec3_t vup(0.0f, 1.0f, 0.0f);
    float vfov = 55.0f;
    float aspect_ratio = 1.0f;
    camera_t camera(lookfrom, lookat, vup, vfov, aspect_ratio);

    // color
    vec3_t red(0.65f, 0.05f, 0.05f);
    vec3_t green(0.12f, 0.45f, 0.15f);
    vec3_t white(0.73f, 0.73f, 0.73f);
    vec3_t brown(0.62f, 0.57f, 0.54f);

    // scene
    scene_t scene;
    scene.spheres.emplace_back(vec3_t(0.5f, 0.2f, -0.25f), 0.2f, brown);
    scene.add_rectangle(vec3_t(0.0f, 0.0f, 0.0f),
                        vec3_t(0.0f, 1.0f, 0.0f),
                        vec3_t(0.0f, 1.0f, -1.0f),
                        vec3_t(0.0f, 0.0f, -1.0f),
                        red);
    scene.add_rectangle(vec3_t(0.0f, 0.0f, -1.0f),
                        vec3_t(0.0f, 1.0f, -1.0f),
                        vec3_t(1.0f, 1.0f, -1.0f),
                        vec3_t(1.0f, 0.0f, -1.0f),
                        white);
    scene.add_rectangle(vec3_t(1.0f, 0.0f, 0.0f),
                        vec3_t(1.0f, 1.0f, 0.0f),
                        vec3_t(1.0f, 1.0f, -1.0f),
                        vec3_t(1.0f, 0.0f, -1.0f),
                        green);
    scene.add_rectangle(vec3_t(0.0f, 1.0f, 0.0f),
                        vec3_t(0.0f, 1.0f, -1.0f),
                        vec3_t(1.0f, 1.0f, -1.0f),
                        vec3_t(1.0f, 1.0f, 0.0f),
                        white);
    scene.add_rectangle(vec3_t(0.0f, 0.0f, 0.0f),
                        vec3_t(0.0f, 0.0f, -1.0f),
                        vec3_t(1.0f, 0.0f, -1.0f),
                        vec3_t(1.0f, 0.0f, 0.0f),
                        white);
    scene.point_light = {vec3_t(0.95f, 0.95f, 0.3f), vec3_t(0.9f, 0.9f, 0.9f)};

    // mpi
    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // custom mpi struct
    vec3_t dummy_vec3{};
    int lengths[3] = { 1, 1, 1 };
    MPI_Aint base_address;
    MPI_Aint displacements[3];
    MPI_Get_address(&dummy_vec3, &base_address);
    MPI_Get_address(&dummy_vec3.v[0], &displacements[0]);
    MPI_Get_address(&dummy_vec3.v[1], &displacements[1]);
    MPI_Get_address(&dummy_vec3.v[2], &displacements[2]);
    displacements[0] = MPI_Aint_diff(displacements[0], base_address);
    displacements[1] = MPI_Aint_diff(displacements[1], base_address);
    displacements[2] = MPI_Aint_diff(displacements[2], base_address);
    MPI_Datatype types[3] = { MPI_FLOAT, MPI_FLOAT, MPI_FLOAT };
    MPI_Datatype vec3_mpi_t;
    MPI_Type_create_struct(3, lengths, displacements, types, &vec3_mpi_t);
    MPI_Type_commit(&vec3_mpi_t);

    // render
    int image_width = 600;
    int image_height = 600;
    int row_per_process = (image_height + world_size - 1) / world_size;
    int start_row = world_rank * row_per_process;
    int expanded_row = world_size * row_per_process;
    vec3_t framebuffer[world_rank == 0 ? expanded_row : row_per_process][image_width];
    MPI_Request requests[world_size - 1];
    if (world_rank == 0) {
        for (int i = 1; i < world_size; i++) {
            MPI_Irecv(&framebuffer[i * row_per_process][0],
                      row_per_process * image_width,
                      vec3_mpi_t,
                      i,
                      0,
                      MPI_COMM_WORLD,
                      &requests[i - 1]);
        }
    }
    for (int i = 0, row = start_row; i < row_per_process && row < image_height; i++, row++) {
        for (int j = 0; j < image_width; j++) {
            float s = float(j) / float(image_width - 1);
            float t = 1.0f - float(row) / float(image_height - 1);
            ray_t camera_ray = camera.get_ray(s, t);
            framebuffer[i][j] = vec3_t::make_zeros();
            for (int k = 1; k <= SAMPLES_PER_PIXEL; k++)
                framebuffer[i][j] = framebuffer[i][j] + get_color(scene, camera_ray);
            framebuffer[i][j] = framebuffer[i][j] / SAMPLES_PER_PIXEL;
        }
    }
    if (world_rank == 0) {
        MPI_Waitall(world_size - 1, requests, MPI_STATUSES_IGNORE);
    } else {
        MPI_Isend(&framebuffer[0][0],
                  row_per_process * image_width,
                  vec3_mpi_t,
                  0,
                  0,
                  MPI_COMM_WORLD,
                  &requests[world_rank - 1]);
    }

    // write framebuffer to file
    if (world_rank == 0) {
        std::ofstream image_fs("image.ppm");
        image_fs << "P3\n" << image_width << ' ' << image_height << "\n255\n";
        for (int i = 0; i < image_height; i++)
            for (int j = 0; j < image_width; j++)
                framebuffer[i][j].write_color(image_fs);
    }

    MPI_Finalize();
    return 0;
}
