# temporarily use the following script to compile: 

NVCC=nvcc

NVCC_OPTS=-arch=sm_35 -rdc=true

GCC_OPTS=-std=c++11

all: make_tests


main:
	$(NVCC) $(GCC_OPTS) main.cu -o main.o -I.

make_tests:
#	$(NVCC) $(NVCC_OPTS) $(GCC_OPTS) tests_tensor.cu -o tests_tensor.o -I.
#	$(NVCC) $(NVCC_OPTS) $(GCC_OPTS) tests_gaussian_initializer.cu -o tests_gaussian_initializer.o -I.
#	$(NVCC) $(NVCC_OPTS) $(GCC_OPTS) tests/tests_data.cu -o tests_data.o -I.
#	$(NVCC) $(NVCC_OPTS) $(GCC_OPTS) tests_conv2d.cu -o tests_conv2d.o -I.
#	$(NVCC) $(NVCC_OPTS) $(GCC_OPTS) tests/tests_softmax.cu -o tests_softmax.o -I.
#	$(NVCC) $(NVCC_OPTS) $(GCC_OPTS) tests/tests_cross_entropy_loss.cu -o tests_cross_entropy_loss.o -I.
	$(NVCC) $(NVCC_OPTS) $(GCC_OPTS) tests/tests_pooling.cu -o tests_pooling.o -I.

clean:
	rm *.o
