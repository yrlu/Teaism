# temporarily use the following script to compile: 

NVCC=nvcc

NVCC_OPTS=-arch=sm_35 -rdc=true

GCC_OPTS=-std=c++11

<<<<<<< HEAD

all: make_tests


main:
	$(NVCC) $(GCC_OPTS) main.cu -o main.o -I.


make_tests:
#	$(NVCC) $(NVCC_OPTS) $(GCC_OPTS) tests_tensor.cu -o tests_tensor.o -I.
#	$(NVCC) $(NVCC_OPTS) $(GCC_OPTS) tests_gaussian_initializer.cu -o tests_gaussian_initializer.o -I.
#	$(NVCC) $(NVCC_OPTS) $(GCC_OPTS) tests_conv2d.cu -o tests_conv2d.o -I.
#	$(NVCC) $(NVCC_OPTS) $(GCC_OPTS) tests/tests_softmax.cu -o tests_softmax.o -I.
	$(NVCC) $(NVCC_OPTS) $(GCC_OPTS) tests/tests_cross_entropy_loss.cu -o tests_cross_entropy_loss.o -I.


=======
all: tests
	
main:
	$(NVCC) $(GCC_OPTS) main.cu -o main.o -I.

tests:
	$(NVCC) $(NVCC_OPTS) $(GCC_OPTS) tests/tests_tensor.cu -o tests_tensor.o -I.
	# $(NVCC) $(NVCC_OPTS) $(GCC_OPTS) tests_gaussian_initializer.cu -o tests_gaussian_initializer.o -I.
	# $(NVCC) $(NVCC_OPTS) $(GCC_OPTS) tests_conv2d.cu -o tests_conv2d.o -I.
>>>>>>> 8dc7ba4ba84758491eea70f4ab08d66b8810d372
clean:
	rm *.o
