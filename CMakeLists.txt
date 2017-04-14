cmake_minimum_required (VERSION 3.5)
project (Teaism)

# The version number.
set (Teaism_VERSION_MAJOR 0)
set (Teaism_VERSION_MINOR 1)

# Add the binary tree to the search path for including files
include_directories("${PROJECT_SOURCE_DIR}")

# Set CXX compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

# Add the executable
add_executable (main.o main.cpp)

