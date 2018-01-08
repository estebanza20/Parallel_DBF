cmake_minimum_required(VERSION 2.8)

project(DBF_GPU)

set(APP_EXEC "dbf_gpu_img")
set(APP_SRC "src/app/dbf_gpu_img.cpp")

#Packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -arch=sm_35)
set(CUDA_SEPARABLE_COMPILATION ON)

include_directories(
  include,
  src,
  ${OPENCV_INCLUDE_DIR},
  ${MPI_CXX_INCLUDE_PATH})

#CUDA
set(CUDA_APP_LIB "dbf_gpu_cuda")
file(GLOB_RECURSE CUDA_SRCS "src/cuda/*.cu")
cuda_add_library(${CUDA_APP_LIB} ${CUDA_SRCS})

#CPP
set(CPP_APP_LIB "dbf_gpu_cpp")
file(GLOB_RECURSE SRCS "src/*.cpp")
add_library(${CPP_APP_LIB} ${SRCS})
target_include_directories(${CPP_APP_LIB} PUBLIC include)

#Executable
add_executable(${APP_EXEC} ${APP_SRC})
target_link_libraries(
  ${APP_EXEC}
  ${CPP_APP_LIB}
  ${CUDA_APP_LIB}
  ${OpenCV_LIBS}
  ${MPI_CXX_LIBRARIES})
