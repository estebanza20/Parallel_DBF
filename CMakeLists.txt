cmake_minimum_required(VERSION 2.8)

project(OpenCVGPU_Test)

set(APP_EXEC "testOCVGPU.x")
set(APP_SRC "tests/ocv_gpu_test.cpp")

#Packages 
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(MPI REQUIRED)


set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_30,code=sm_30)

include_directories(
  include,
  src,
  ${OPENCV_INCLUDE_DIR},
  ${MPI_CXX_INCLUDE_PATH})

#CUDA Related
file(GLOB CUDA_SRCS "src/cuda/*.cu")
cuda_add_library(CudaFunctions ${CUDA_SRCS})

#CPP Related
file(GLOB SRCS "src/*.cpp")
add_library(TestOpenCVGPU ${SRCS})
target_include_directories(TestOpenCVGPU PUBLIC include)

#Executable
add_executable(${APP_EXEC} ${APP_SRC})
target_link_libraries(
  ${APP_EXEC}
  TestOpenCVGPU
  CudaFunctions
  ${OpenCV_LIBS}
  ${MPI_CXX_LIBRARIES})
