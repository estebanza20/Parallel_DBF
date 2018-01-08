cmake_minimum_required(VERSION 2.8)

project(DBF_GPU)

set(APP_EXEC "dbf_gpu_img.x")
set(APP_SRC "app/dbf_gpu_img.cpp")

#Packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(MPI REQUIRED)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_30,code=sm_30)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -arch=sm_35)
set(CUDA_SEPARABLE_COMPILATION ON)

include_directories(
  include,
  src,
  ${OPENCV_INCLUDE_DIR},
  ${MPI_CXX_INCLUDE_PATH})

#CUDA Related
# file(GLOB CUDA_SRCS "src/cuda/*.cu")
file(GLOB_RECURSE CUDA_SRCS "src/cuda/*.cu")
cuda_add_library(CudaRelated ${CUDA_SRCS})

#CPP Related
# file(GLOB SRCS "src/*.cpp")
file(GLOB_RECURSE SRCS "src/*.cpp")
add_library(CppRelated ${SRCS})
target_include_directories(CppRelated PUBLIC include)

#Executable
add_executable(${APP_EXEC} ${APP_SRC})
target_link_libraries(
  ${APP_EXEC}
  CppRelated
  CudaRelated
  ${OpenCV_LIBS}
  ${MPI_CXX_LIBRARIES})
