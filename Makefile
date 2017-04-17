# temporarily use the following script to compile: 

NVCC=nvcc

GCC_OPTS=-std=c++11

all: gaussian.o conv2d.o main

gaussian.o:
	$(NVCC) -c $(GCC_OPTS) initializers/gaussian_kernel_initializer.cu -I. 

conv2d.o:
	$(NVCC) -c $(GCC_OPTS) layers/conv2d.cu -I.

# main: gaussian.o conv2d.o
main:
	$(NVCC) $(GCC_OPTS) main.cu -o main.o -I.

clean:
	rm *.o