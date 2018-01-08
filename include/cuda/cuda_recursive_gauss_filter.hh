#ifndef _CUDA_RECURSIVE_GAUSS_FILTER_H_
#define _CUDA_RECURSIVE_GAUSS_FILTER_H_

// CUDA related includes
#include "cuda/cuda_common.hh"
#include "cuda/cuda_util.hh"
#include "cuda/cuda_transpose.hh"

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

using namespace cv::cuda;

#define MAX_EPSILON_ERROR 5.0f
#define THRESHOLD  0.15f

void gaussianFilter_RGB_GPU(const GpuMat& d_src, GpuMat& d_dest, GpuMat& d_temp,
                            float sigma, int order, int nthreads);

#endif
