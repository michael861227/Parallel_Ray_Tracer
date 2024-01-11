# Parallel Ray Tracer

[Project Report](report.pdf)

[Presentation Slides](slides.pdf)

<BR>

## Build
```shell
make
```

<BR>

## Prerequisite for MPI
### 0. Initial Setting
```shell
1. mkdir -p ~/.ssh
2. ssh-keygen -t rsa # Leave all empty
```

### 1. Copy the config to `~/.ssh/config`
```Shell
3. cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
```

### 2. Enter pp2 to pp10
```Shell
ssh pp2
ssh pp3
.
.
.
ssh pp10
```

### 3. Maintain consistency by copying the data from the `.ssh` directory, ensuring that the keys on each computer are uniform.
```shell
4. parallel-scp -A -h host.txt -r ~/.ssh ~
```

<BR>

## Execute Each Programming Model

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
parallel-scp -h host.txt mpi_exe ~
time mpirun -np 8 --hostfile host.txt ./mpi_exe
```

### CUDA Mega
```shell
time ./cuda_b_exe
```

### CUDA Split
```shell
time ./cuda_c_exe
```
