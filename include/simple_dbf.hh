
#ifndef _SIMPLE_CUDA_DBF_H_
#define _SIMPLE_CUDA_DBF_H_

// System Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// CUDA includes and interop headers
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_math.h>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

using namespace cv::cuda;

int iDivUp(int a, int b);

void simpleDBF_RGB_GPU(const GpuMat& d_src,
                       GpuMat& d_dest,
                       int kernel_size,
                       float sigma_color,
                       float sigma_space);

#endif
