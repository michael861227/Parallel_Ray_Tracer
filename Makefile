all: serial_exe cuda_exe

serial_exe:
	g++ -O3 -std=c++17 serial/main.cpp -o serial_exe

cuda_exe:
	nvcc -O3 -std=c++17 cuda/main.cu -o cuda_exe

clean:
	rm -f serial_exe cuda_exe image.ppm
