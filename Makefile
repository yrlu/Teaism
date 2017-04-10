# TODO: CMakelist
# temporarily use the following script to compile: (without CUDA)
all: main

main:
	g++ -std=c++11 main.cpp -o main.o -I.

clean: 
	rm *.o