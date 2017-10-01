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
*/

#ifndef _RECURSIVEGAUSSIAN_KERNEL_CU_
#define _RECURSIVEGAUSSIAN_KERNEL_CU_

// System Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// CUDA includes and interop headers
#include <helper_cuda.h>
#include <helper_math.h>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

using namespace cv::cuda;

#define BLOCK_DIM 16
#define CLAMP_TO_EDGE 1


// Transpose kernel (see transpose CUDA Sample for details)
__global__ void d_transpose(const PtrStepSz<uchar3> src, PtrStepSz<uchar3> dest)
{
   __shared__ uchar3 block[BLOCK_DIM][BLOCK_DIM+1];

   // read the matrix tile into shared memory
   unsigned int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
   unsigned int y = blockIdx.y * BLOCK_DIM + threadIdx.y;

   int width = src.cols;
   int height = src.rows;
   
   if ((x < width) && (y < height))
   {
      block[threadIdx.y][threadIdx.x] = src(y,x);
   }

   __syncthreads();

   // write the transposed matrix tile to global memory
   x = blockIdx.y * BLOCK_DIM + threadIdx.x;
   y = blockIdx.x * BLOCK_DIM + threadIdx.y;

   if ((x < height) && (y < width))
   {
      dest(y,x) = block[threadIdx.x][threadIdx.y];
   }
}



__device__ float3 gpuMatElemToFloat(const uchar3 elem)
{
   float3 r;
   r.x = elem.x/255.0f;
   r.y = elem.y/255.0f;
   r.z = elem.z/255.0f;
   //r.w = 0.0;
   return r;
}

__device__ uchar3 floatToGpuMatElem(const float3 val)
{
   uchar3 r;
   r.x = __saturatef(val.x)*255;
   r.y = __saturatef(val.y)*255;
   r.z = __saturatef(val.z)*255;
   return r;
}


__global__ void
d_recursiveGaussian_rgba(const PtrStepSz<uchar3> src, PtrStepSz<uchar3> dest,
			 int w, int h,
			 float a0, float a1, float a2, float a3,
			 float b1, float b2,
			 float coefp, float coefn)
{
   unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

   if (x >= w) return;

   // forward pass
   float3 xp = make_float3(0.0f);  // previous input
   float3 yp = make_float3(0.0f);  // previous output
   float3 yb = make_float3(0.0f);  // previous output by 2
#if CLAMP_TO_EDGE
   xp = gpuMatElemToFloat(src(0,x));
   yb = coefp*xp;
   yp = yb;
#endif

   for (int y = 0; y < h; y++)
   {
      float3 xc = gpuMatElemToFloat(src(y,x));
      float3 yc = a0*xc + a1*xp - b1*yp - b2*yb;
      dest(y,x) = floatToGpuMatElem(yc);
      xp = xc;
      yb = yp;
      yp = yc;
   }

   // reverse pass
   // ensures response is symmetrical
   float3 xn = make_float3(0.0f);
   float3 xa = make_float3(0.0f);
   float3 yn = make_float3(0.0f);
   float3 ya = make_float3(0.0f);
#if CLAMP_TO_EDGE
   xn = xa = gpuMatElemToFloat(src(h-1,x));
   yn = coefn*xn;
   ya = yn;
#endif

   for (int y = h-1; y >= 0; y--)
   {
      float3 xc = gpuMatElemToFloat(src(y,x));
      float3 yc = a2*xn + a3*xa - b1*yn - b2*ya;
      xa = xn;
      xn = xc;
      ya = yn;
      yn = yc;
      dest(y,x) = floatToGpuMatElem(gpuMatElemToFloat(dest(y,x)) + yc);
   }
}

#endif
