/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Recursive Gaussian filter
    sgreen 8/1/08

    This code sample implements a Gaussian blur using Deriche's recursive method:
    http://citeseer.ist.psu.edu/deriche93recursively.html

    This is similar to the box filter sample in the SDK, but it uses the previous
    outputs of the filter as well as the previous inputs. This is also known as an
    IIR (infinite impulse response) filter, since its response to an input impulse
    can last forever.

    The main advantage of this method is that the execution time is independent of
    the filter width.

    The GPU processes columns of the image in parallel. To avoid uncoalesced reads
    for the row pass we transpose the image and then transpose it back again
    afterwards.

    The implementation is based on code from the CImg library:
    http://cimg.sourceforge.net/
    Thanks to David Tschumperl� and all the CImg contributors!
*/


// CUDA includes and interop headers
#include <cuda_runtime.h>

// CUDA utilities and system includes
#include <helper_cuda.h>      // includes cuda.h and cuda_runtime_api.h
#include <helper_functions.h>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

using namespace cv;

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define MAX_EPSILON_ERROR 5.0f
#define THRESHOLD  0.15f


float sigma = 10.0f;
int order = 0;
int nthreads = 64;  // number of threads per block

unsigned int width, height;
unsigned int *h_img = NULL;
unsigned int *d_img = NULL;
unsigned int *d_temp = NULL;

StopWatchInterface *timer = 0;
int *pArgc = NULL;
char **pArgv = NULL;

bool runBenchmark = false;

const char *sSDKsample = "CUDA Recursive Gaussian";

extern "C"
void transpose(unsigned int *d_src, unsigned int *d_dest, unsigned int width, int height);

extern "C"
void gaussianFilterRGBA(unsigned int *d_src, unsigned int *d_dest, unsigned int *d_temp, int width, int height, float sigma, int order, int nthreads);


void gaussianFilterRGBA(const cuda::GpuMat& d_src,
			cuda::GpuMat& d_dest,
			cuda::GpuMat& d_temp,
			float sigma,
			int order,
			int nthreads);


void cleanup();

void initCudaBuffers()
{
    unsigned int size = width * height * sizeof(unsigned int);

    // allocate device memory
    checkCudaErrors(cudaMalloc((void **) &d_img, size));
    checkCudaErrors(cudaMalloc((void **) &d_temp, size));

    checkCudaErrors(cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice));

    sdkCreateTimer(&timer);
}


void benchmark(int iterations)
{
    // allocate memory for result
    unsigned int *d_result;
    unsigned int size = width * height * sizeof(unsigned int);
    checkCudaErrors(cudaMalloc((void **) &d_result, size));

    // warm-up
    gaussianFilterRGBA(d_img, d_result, d_temp, width, height, sigma, order, nthreads);

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStartTimer(&timer);

    // execute the kernel
    for (int i = 0; i < iterations; i++)
    {
        gaussianFilterRGBA(d_img, d_result, d_temp, width, height, sigma, order, nthreads);
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);

    // check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n", (width*height*iterations / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);

    checkCudaErrors(cudaFree(d_result));
}

void kernel_test(const cuda::GpuMat& d_src, cuda::GpuMat& d_dest, float sigma, int order)
{
   int nthreads = 64;
   StopWatchInterface *timer0 = 0;
   sdkCreateTimer(&timer0);

   cuda::GpuMat d_temp;
   d_temp.create(d_src.rows, d_src.cols, d_src.type());
   
   std::cout << "Kernel test running: \n";

   sdkStartTimer(&timer0);
   gaussianFilterRGBA(d_src, d_dest, d_temp, sigma, order, nthreads);
   sdkStopTimer(&timer0);
   
   printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer0));
}
