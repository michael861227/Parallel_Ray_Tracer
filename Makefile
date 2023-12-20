all: serial_exe cuda_a_exe

serial_exe: FORCE
	g++ -O3 -std=c++17 -Wall serial/main.cpp -o serial_exe

cuda_a_exe: FORCE
	nvcc -O3 -std=c++17 -arch=native cuda/main.cu -o cuda_a_exe

clean: FORCE
	rm -f serial_exe cuda_a_exe image.ppm

FORCE: