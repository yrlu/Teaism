# temporarily use the following script to compile: 
all: main

main:
	nvcc -std=c++11 main.cpp -o main.o -I.

clean: 
	rm *.o