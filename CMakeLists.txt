cmake_minimum_required(VERSION 3.22)

project(PSTANN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

find_library(PROFILER_LIB profiler REQUIRED)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    
else()
    message(FATAL_ERROR "OpenMP not found")
endif()

SET(CMAKE_CXX_FLAGS "-O3 -march=native -lrt -DHAVE_CXX0X -fpic -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 -ffast-math" )

add_subdirectory(tests)
add_subdirectory(utils)
