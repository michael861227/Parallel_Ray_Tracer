all: serial_exe cuda_a_exe cuda_b_exe

serial_exe: FORCE
	g++ -O3 -std=c++17 -Wall serial/main.cpp -o serial_exe

cuda_a_exe: FORCE
	nvcc -O3 -std=c++17 -arch=native cuda_a/main.cu -o cuda_a_exe

cuda_b_exe: FORCE
	nvcc -O3 -std=c++17 -arch=native cuda_b/main.cu -o cuda_b_exe

clean: FORCE
	rm -f serial_exe cuda_a_exe cuda_b_exe image.ppm

FORCE: