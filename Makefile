# temporarily use the following script to compile: 

NVCC=nvcc

NVCC_OPTS=-arch=sm_35 -rdc=true

GCC_OPTS=-std=c++11

all: gaussian.o conv2d.o main

gaussian.o:
	$(NVCC) -c $(GCC_OPTS) initializers/gaussian_kernel_initializer.cu -I. 

conv2d.o:
	$(NVCC) -c $(GCC_OPTS) layers/conv2d.cu -I.

# main: gaussian.o conv2d.o
main:
	$(NVCC) $(GCC_OPTS) main.cu -o main.o -I.

tests:
	$(NVCC) $(NVCC_OPTS) $(GCC_OPTS) tests_tensor.cu -o tests_tensor.o -I.
	# $(NVCC) $(GCC_OPTS) tests_gaussian_initializer.cu -o tests_gaussian_initializer.o -I.
	$(NVCC) $(NVCC_OPTS) $(GCC_OPTS) tests_conv2d.cu -o tests_conv2d.o -I.
clean:
	rm *.o