all: serial_exe

serial_exe:
	g++ -o serial_exe -O3 serial/main.cpp

clean:
	rm serial_exe image.ppm