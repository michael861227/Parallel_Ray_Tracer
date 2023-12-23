#ifndef PARALLEL_RAY_TRACER_RENDER_H
#define PARALLEL_RAY_TRACER_RENDER_H

#include <cub/cub.cuh>

// use structure-of-array for memory coalescing
struct ray_pool_t {
    int pixel_idx[2 * NUM_WORKING_PATHS];
    ray_t ray[2 * NUM_WORKING_PATHS];
};

struct path_ray_payload_t {
    bool hit[NUM_WORKING_PATHS];
    record_t record[NUM_WORKING_PATHS];
    int bounces[NUM_WORKING_PATHS];
    vec3_t multiplier[NUM_WORKING_PATHS];
};

struct shadow_ray_payload_t {
    vec3_t color[NUM_WORKING_PATHS];
};

__constant__ camera_t* d_camera;
__constant__ scene_t* d_scene;
__constant__ vec3_t* d_framebuffer;
__constant__ curandState* d_rand_states;
__constant__ ray_pool_t* d_ray_pool;
__constant__ path_ray_payload_t* d_path_ray_payload;
__constant__ shadow_ray_payload_t* d_shadow_ray_payload;

__constant__ bool* d_color_pending_valid;
__constant__ bool* d_gen_pending_valid;
__constant__ bool* d_shit_pending_valid;
__constant__ bool* d_phit_pending_valid;

__constant__ int* d_color_pending;
__constant__ int* d_gen_pending;
__constant__ int* d_shit_pending;
__constant__ int* d_phit_pending;

__constant__ int* d_color_pending_compact;
__constant__ int* d_gen_pending_compact;
__constant__ int* d_shit_pending_compact;
__constant__ int* d_phit_pending_compact;

__constant__ int* d_num_color_pending;
__constant__ int* d_num_gen_pending;
__constant__ int* d_num_shit_pending;
__constant__ int* d_num_phit_pending;

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

__global__ void init_framebuffer() {
    int thread_id = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (thread_id >= NUM_PIXELS)
        return;
    d_framebuffer[thread_id] = vec3_t::make_zeros();
}

__global__ void init_rand_states() {
    int thread_id = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (thread_id >= NUM_WORKING_PATHS)
        return;
    curand_init(RAND_SEED, thread_id, 0, &d_rand_states[thread_id]);
}

__global__ void init_path_ray_payload() {
    int thread_id = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (thread_id >= NUM_WORKING_PATHS)
        return;
    d_path_ray_payload->bounces[thread_id] = INT_MAX;
}

__global__ void logic() {
    int thread_id = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (thread_id >= NUM_WORKING_PATHS)
        return;

    d_color_pending_valid[thread_id] = false;
    d_gen_pending_valid[thread_id] = false;
    d_shit_pending_valid[thread_id] = false;
    d_phit_pending_valid[thread_id] = false;
    d_phit_pending_valid[NUM_WORKING_PATHS + thread_id] = false;

    int &bounces = d_path_ray_payload->bounces[thread_id];
    bool continue_path = bounces < MAX_PATH;
    bounces++;

    if (continue_path && d_path_ray_payload->hit[thread_id]) {
        d_color_pending_valid[thread_id] = true;
        d_color_pending[thread_id] = thread_id;
    } else {
        d_gen_pending_valid[thread_id] = true;
        d_gen_pending[thread_id] = thread_id;
    }
}

__global__ void color() {
    int thread_id = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (thread_id >= *d_num_color_pending)
        return;

    int path_ray_id = d_color_pending_compact[thread_id];
    int pixel_idx = d_ray_pool->pixel_idx[path_ray_id];
    ray_t ray = d_ray_pool->ray[path_ray_id];

    record_t record = d_path_ray_payload->record[path_ray_id];

    // generate next ray
    int &bounces = d_path_ray_payload->bounces[path_ray_id];
    vec3_t &multiplier = d_path_ray_payload->multiplier[path_ray_id];
    multiplier = multiplier * record.albedo;
    curandState &rand_state = d_rand_states[path_ray_id];
    if (bounces < MAX_PATH) {
        d_ray_pool->ray[path_ray_id] = {record.hit_point,
                                        record.unit_n + vec3_t::uniform_sample_sphere(rand_state),
                                        EPS,
                                        FLT_MAX};
        d_phit_pending_valid[NUM_WORKING_PATHS + thread_id] = true;
        d_phit_pending[NUM_WORKING_PATHS + thread_id] = path_ray_id;
    }

    // generate shadow ray
    vec3_t shadow_dir = d_scene->point_light.position - record.hit_point;
    if (dot(ray.direction, record.unit_n) * dot(shadow_dir, record.unit_n) < 0.f) {  // in same hemisphere
        int shadow_ray_id = NUM_WORKING_PATHS + path_ray_id;
        d_ray_pool->pixel_idx[shadow_ray_id] = pixel_idx;
        d_ray_pool->ray[shadow_ray_id] = {record.hit_point, shadow_dir, EPS, 1.0f};
        float t2 = shadow_dir.length_squared();
        float t = std::sqrt(t2);
        d_shadow_ray_payload->color[path_ray_id] = multiplier * d_scene->point_light.intensity / t2 *
                                                   dot(shadow_dir, record.unit_n) / t;  // cos(theta)
        d_shit_pending_valid[thread_id] = true;
        d_shit_pending[thread_id] = shadow_ray_id;
    }
}

__global__ void gen(int camera_ray_start_id) {
    int thread_id = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (thread_id >= *d_num_gen_pending)
        return;
    int camera_ray_id = camera_ray_start_id + thread_id;
    if (camera_ray_id >= NUM_PIXELS * SAMPLES_PER_PIXEL)
        return;

    int pixel_idx = camera_ray_id / SAMPLES_PER_PIXEL;
    int i = pixel_idx % IMAGE_WIDTH;
    int j = pixel_idx / IMAGE_WIDTH;
    int path_ray_id = d_gen_pending_compact[thread_id];

    float s = float(i) / float(IMAGE_WIDTH - 1);
    float t = 1.0f - float(j) / float(IMAGE_HEIGHT - 1);
    d_ray_pool->pixel_idx[path_ray_id] = pixel_idx;
    d_ray_pool->ray[path_ray_id] = d_camera->get_ray(s, t);
    d_path_ray_payload->bounces[path_ray_id] = 0;
    d_path_ray_payload->multiplier[path_ray_id] = vec3_t::make_ones();

    d_phit_pending_valid[thread_id] = true;
    d_phit_pending[thread_id] = path_ray_id;
}

__global__ void shit() {
    int thread_id = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (thread_id >= *d_num_shit_pending)
        return;

    int shadow_ray_id = d_shit_pending_compact[thread_id];
    int path_ray_id = shadow_ray_id - NUM_WORKING_PATHS;
    ray_t ray = d_ray_pool->ray[shadow_ray_id];
    if (!occluded(*d_scene, ray))
        d_framebuffer[d_ray_pool->pixel_idx[shadow_ray_id]].atomic_add(d_shadow_ray_payload->color[path_ray_id]);
}

__global__ void phit() {
    int thread_id = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (thread_id >= *d_num_phit_pending)
        return;

    int path_ray_id = d_phit_pending_compact[thread_id];
    ray_t ray = d_ray_pool->ray[path_ray_id];

    bool hit = false;
    record_t record{};
    for (int i = 0; i < d_scene->num_spheres; i++)
        if (d_scene->spheres[i].hit(ray, record))
            hit = true;
    for (int i = 0; i < d_scene->num_trigs; i++)
        if (d_scene->trigs[i].hit(ray, record))
            hit = true;

    // save intersection result
    d_path_ray_payload->hit[path_ray_id] = hit;
    d_path_ray_payload->record[path_ray_id] = record;
}

void compact(int num_items, bool* d_flags, int* d_in, int* d_out, int* d_num_selected_out) {
    static void* d_temp_storage = nullptr;
    static size_t temp_storage_bytes = 0;

    size_t new_temp_storage_bytes;
    CHECK_CUDA(cub::DeviceSelect::Flagged(nullptr, new_temp_storage_bytes, d_in, d_flags, d_out,
                                          d_num_selected_out, num_items));

    if (new_temp_storage_bytes > temp_storage_bytes) {
        if (d_temp_storage != nullptr)
            CHECK_CUDA(cudaFree(d_temp_storage));
        CHECK_CUDA(cudaMalloc(&d_temp_storage, new_temp_storage_bytes));
        temp_storage_bytes = new_temp_storage_bytes;
    }

    CHECK_CUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out,
                                          d_num_selected_out, num_items));
}

template <typename T>
T* cuda_malloc_symbol(T* &symbol, const size_t size) {
    T* tmp;
    CHECK_CUDA(cudaMalloc(&tmp, size));
    CHECK_CUDA(cudaMemcpyToSymbol(symbol, &tmp, sizeof(T*)));
    return tmp;
}

void render(const camera_t* d_camera_ptr, const scene_t* d_scene_ptr, const vec3_t* d_framebuffer_ptr) {
    CHECK_CUDA(cudaMemcpyToSymbol(d_camera, &d_camera_ptr, sizeof(camera_t*)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_scene, &d_scene_ptr, sizeof(scene_t*)));
    CHECK_CUDA(cudaMemcpyToSymbol(d_framebuffer, &d_framebuffer_ptr, sizeof(vec3_t*)));
    cuda_malloc_symbol(d_rand_states, NUM_WORKING_PATHS * sizeof(curandState));

    bool* d_color_pending_valid_ptr = cuda_malloc_symbol(d_color_pending_valid, NUM_WORKING_PATHS * sizeof(bool));
    bool* d_gen_pending_valid_ptr = cuda_malloc_symbol(d_gen_pending_valid, NUM_WORKING_PATHS * sizeof(bool));
    bool* d_shit_pending_valid_ptr = cuda_malloc_symbol(d_shit_pending_valid, NUM_WORKING_PATHS * sizeof(bool));
    bool* d_phit_pending_valid_ptr = cuda_malloc_symbol(d_phit_pending_valid, 2 * NUM_WORKING_PATHS * sizeof(bool));

    int* d_color_pending_ptr = cuda_malloc_symbol(d_color_pending, NUM_WORKING_PATHS * sizeof(int));
    int* d_gen_pending_ptr = cuda_malloc_symbol(d_gen_pending, NUM_WORKING_PATHS * sizeof(int));
    int* d_shit_pending_ptr = cuda_malloc_symbol(d_shit_pending, NUM_WORKING_PATHS * sizeof(int));
    int* d_phit_pending_ptr = cuda_malloc_symbol(d_phit_pending, 2 * NUM_WORKING_PATHS * sizeof(int));

    int* d_color_pending_compact_ptr = cuda_malloc_symbol(d_color_pending_compact, NUM_WORKING_PATHS * sizeof(int));
    int* d_gen_pending_compact_ptr = cuda_malloc_symbol(d_gen_pending_compact, NUM_WORKING_PATHS * sizeof(int));
    int* d_shit_pending_compact_ptr = cuda_malloc_symbol(d_shit_pending_compact, NUM_WORKING_PATHS * sizeof(int));
    int* d_phit_pending_compact_ptr = cuda_malloc_symbol(d_phit_pending_compact, 2 * NUM_WORKING_PATHS * sizeof(int));

    int* d_num_color_pending_ptr = cuda_malloc_symbol(d_num_color_pending, sizeof(int));
    int* d_num_gen_pending_ptr = cuda_malloc_symbol(d_num_gen_pending, sizeof(int));
    int* d_num_shit_pending_ptr = cuda_malloc_symbol(d_num_shit_pending, sizeof(int));
    int* d_num_phit_pending_ptr = cuda_malloc_symbol(d_num_phit_pending, sizeof(int));

    cuda_malloc_symbol(d_ray_pool, sizeof(ray_pool_t));
    cuda_malloc_symbol(d_path_ray_payload, sizeof(path_ray_payload_t));
    cuda_malloc_symbol(d_shadow_ray_payload, sizeof(shadow_ray_payload_t));

    init_framebuffer<<<(NUM_PIXELS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>();
    CHECK_CUDA(cudaGetLastError());
    init_rand_states<<<(NUM_WORKING_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>();
    CHECK_CUDA(cudaGetLastError());
    init_path_ray_payload<<<(NUM_WORKING_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>();
    CHECK_CUDA(cudaGetLastError());

    int num_color_pending;
    int num_gen_pending;
    int num_shit_pending;
    int num_phit_pending;
    int camera_ray_start_id = 0;
    while (true) {
        logic<<<(NUM_WORKING_PATHS + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>();
        CHECK_CUDA(cudaGetLastError());

        compact(NUM_WORKING_PATHS, d_color_pending_valid_ptr, d_color_pending_ptr,
                d_color_pending_compact_ptr, d_num_color_pending_ptr);
        compact(NUM_WORKING_PATHS, d_gen_pending_valid_ptr, d_gen_pending_ptr,
                d_gen_pending_compact_ptr, d_num_gen_pending_ptr);
        CHECK_CUDA(cudaMemcpy(&num_color_pending, d_num_color_pending_ptr, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&num_gen_pending, d_num_gen_pending_ptr, sizeof(int), cudaMemcpyDeviceToHost));
        if (num_color_pending == 0 && camera_ray_start_id >= NUM_PIXELS * SAMPLES_PER_PIXEL)
            break;

        if (num_color_pending > 0) {
            color<<<(num_color_pending + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>();
            CHECK_CUDA(cudaGetLastError());
        }

        if (num_gen_pending > 0) {
            gen<<<(num_gen_pending + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(camera_ray_start_id);
            CHECK_CUDA(cudaGetLastError());
        }
        camera_ray_start_id += num_gen_pending;

        compact(NUM_WORKING_PATHS, d_shit_pending_valid_ptr, d_shit_pending_ptr,
                d_shit_pending_compact_ptr, d_num_shit_pending_ptr);
        compact(2 * NUM_WORKING_PATHS, d_phit_pending_valid_ptr, d_phit_pending_ptr,
                d_phit_pending_compact_ptr, d_num_phit_pending_ptr);
        cudaMemcpy(&num_shit_pending, d_num_shit_pending_ptr, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&num_phit_pending, d_num_phit_pending_ptr, sizeof(int), cudaMemcpyDeviceToHost);

        if (num_shit_pending > 0) {
            shit<<<(num_shit_pending + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>();
            CHECK_CUDA(cudaGetLastError());
        }

        if (num_phit_pending > 0) {
            phit<<<(num_phit_pending + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>();
            CHECK_CUDA(cudaGetLastError());
        }
    }
}

#endif //PARALLEL_RAY_TRACER_RENDER_H
