cmake_minimum_required(VERSION 3.0)
project(parallel_ray_tracer)

set(CMAKE_CXX_STANDARD 17)

add_executable(parallel_ray_tracer serial/main.cpp)
