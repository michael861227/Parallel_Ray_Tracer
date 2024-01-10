# parallel-ray-tracer

## Build
```shell
make
```

## Execute

### SIMD
```shell
time ./simd_exe
```

### OpenMP
```shell
time ./openmp_exe
```

### MPI
```shell
time mpirun ./mpi_exe
```

### CUDA Mega
```shell
time ./cuda_b_exe
```

### CUDA Split
```shell
time ./cuda_c_exe
```