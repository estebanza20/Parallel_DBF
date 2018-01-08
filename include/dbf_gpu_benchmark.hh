#ifndef _CUDA_DBF_BENCHMARK_H_
#define _CUDA_DBF_BENCHMARK_H_

#include "cuda/cuda_dbf.hh"

void dbf_RGB_GPU_simpleBenchmark(const GpuMat& d_src, GpuMat& d_dest,
                                  float kernel_size, float sigma_color,
                                  float sigma_space, int iterations);

void dbf_RGB_GPU_shmemBenchmark(const GpuMat& d_src, GpuMat& d_dest,
                                  float kernel_size, float sigma_color,
                                  float sigma_space, int iterations);

void dbf_RGB_GPU_texBenchmark(const GpuMat& d_src, GpuMat& d_dest,
                              float kernel_size, float sigma_color,
                              float sigma_space, int iterations);

#endif
