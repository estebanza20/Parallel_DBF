cmake_minimum_required(VERSION 2.8)

project(Simple_CUDA_DBF)

set(APP_EXEC "simple_cuda_dbf.x")
set(APP_SRC "tests/simple_cuda_dbf.cpp")

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
cuda_add_library(CudaRelated ${CUDA_SRCS})

#CPP Related
file(GLOB SRCS "src/*.cpp")
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
