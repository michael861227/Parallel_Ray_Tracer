all: serial_exe cuda_a_exe cuda_b_exe cuda_c_exe mpi_exe dummy

serial_exe: FORCE
	g++ -O3 -std=c++17 -Wall serial/main.cpp -o serial_exe

cuda_a_exe: FORCE
	nvcc -O3 -std=c++17 -arch=native cuda_a/main.cu -o cuda_a_exe

cuda_b_exe: FORCE
	nvcc -O3 -std=c++17 -arch=native cuda_b/main.cu -o cuda_b_exe

cuda_c_exe: FORCE
	nvcc -O3 -std=c++17 -arch=native cuda_c/main.cu -o cuda_c_exe

mpi_exe: FORCE
	mpicxx -O3 -std=c++17 -Wall mpi/main.cpp -o mpi_exe

dummy: FORCE
	g++ mpi/main.cpp -o /dev/null

clean: FORCE
	rm -f serial_exe cuda_a_exe cuda_b_exe cuda_c_exe mpi_exe image.ppm

FORCE: