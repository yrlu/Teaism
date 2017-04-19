cmake_minimum_required (VERSION 3.5)
project (Teaism)

# The version number.
set (Teaism_VERSION_MAJOR 0)
set (Teaism_VERSION_MINOR 1)

# Add the binary tree to the search path for including files
include_directories("${PROJECT_SOURCE_DIR}")

# Set CUDA

find_package(CUDA QUIET REQUIRED)

# Set CXX compiler / CUDA compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

# Add the executable
# cuda_add_executable (main.o main.cu)
# cuda_add_executable (tests_data.o tests/tests_data.cu)
cuda_add_executable (tests_softmax.o tests/tests_softmax.cu)

