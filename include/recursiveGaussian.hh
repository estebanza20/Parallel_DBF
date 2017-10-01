#ifndef _RECURSIVEGAUSSIAN_H_
#define _RECURSIVEGAUSSIAN_H_

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

#define MAX_EPSILON_ERROR 5.0f
#define THRESHOLD  0.15f


int iDivUp(int a, int b);

void transpose(const GpuMat& d_src, GpuMat& d_dest);

void gaussianFilterRGBA(const GpuMat& d_src, GpuMat& d_dest, GpuMat& d_temp,
			float sigma, int order, int nthreads);

#endif
