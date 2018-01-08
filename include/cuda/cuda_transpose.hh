#ifndef _CUDA_TRANSPOSE_H_
#define _CUDA_TRANSPOSE_H_

// CUDA related includes
#include "cuda/cuda_common.hh"
#include "cuda/cuda_util.hh"

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

using namespace cv::cuda;

#define BLOCK_DIM 16

void transpose_GPU(const GpuMat& d_src, GpuMat& d_dest);

#endif
