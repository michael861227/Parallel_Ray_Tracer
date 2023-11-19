all: serial_exe

serial_exe:
	g++ -o serial_exe -O3 -std=c++17 serial/main.cpp

clean:
	rm -f serial_exe image.ppm