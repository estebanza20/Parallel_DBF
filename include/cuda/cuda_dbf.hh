#ifndef _CUDA_DBF_H_
#define _CUDA_DBF_H_

// CUDA related includes
#include "cuda/cuda_common.hh"
#include "cuda/cuda_util.hh"

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

using namespace cv::cuda;

// DBF implementations
void dbf_RGB_GPU_simple(const GpuMat& d_src, GpuMat& d_dest, int kernel_size,
                       float sigma_color, float sigma_space);

void dbf_RGB_GPU_shmem(const GpuMat& d_src, GpuMat& d_dest, int kernel_size,
                         float sigma_color, float sigma_space);

void dbf_RGB_GPU_tex(const GpuMat& d_src, GpuMat& d_dest, int kernel_size,
                     float sigma_color, float sigma_space);

#endif
