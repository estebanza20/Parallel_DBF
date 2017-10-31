#ifndef _SIMPLE_CUDA_DBF_KERNEL_CU_
#define _SIMPLE_CUDA_DBF_KERNEL_CU_

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


__device__ __forceinline__ float norm_l1(const float& a)  { return ::fabs(a); }
__device__ __forceinline__ float norm_l1(const float2& a) { return ::fabs(a.x) + ::fabs(a.y); }
__device__ __forceinline__ float norm_l1(const float3& a) { return ::fabs(a.x) + ::fabs(a.y) + ::fabs(a.z); }
__device__ __forceinline__ float norm_l1(const float4& a) { return ::fabs(a.x) + ::fabs(a.y) + ::fabs(a.z) + ::fabs(a.w); }

// __device__ __forceinline__ float sqr(const float& a) {return a * a;}

__device__ __forceinline__ float norm_l2(const float2& a) { return ::dot(a,a);}
__device__ __forceinline__ float norm_l2(const float3& a) { return ::dot(a,a);}
__device__ __forceinline__ float norm_l2(const float4& a) { return ::dot(a,a);}
//
// __device__ __forceinline__ float norm_l2(const float& a)  { return a.x * a.x}
// __device__ __forceinline__ float norm_l2(const float2& a) { return a.x * a.x + a.y * a.y }
// __device__ __forceinline__ float norm_l2(const float3& a) { return a.x * a.x + a.y * a.y + a.z * a.z}
// __device__ __forceinline__ float norm_l2(const float4& a) { return a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w}


__device__
float3 gpuMatElemToFloat(const uchar3 elem)
{
   float3 r;
   r.x = elem.x/255.0f;
   r.y = elem.y/255.0f;
   r.z = elem.z/255.0f;
   return r;
}

__device__
uchar3 floatToGpuMatElem(const float3 val)
{
   uchar3 r;
   r.x = __saturatef(val.x)*255;
   r.y = __saturatef(val.y)*255;
   r.z = __saturatef(val.z)*255;
   return r;
}


__global__
void d_simpleDBF_RGB(const PtrStepSz<uchar3> src, PtrStep<uchar3> src_sharp, PtrStep<uchar3> dest,
                     const int k_size, const float sigma_spatial2_inv_half, const float sigma_color2_inv_half)
{
  int height = src.rows;
  int width = src.cols;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  float3 color_sum = make_float3(0.0f);
  float norm_sum = 0.0f;

  int r = k_size/2;

  float3 pixel_color;
  float3 pixel_color_sharp;
  float3 center_color = gpuMatElemToFloat(src(y,x));

  float weight;
  float space_dist2;

  int xc, yc;


  if (x-r >=0 && y-r >=0 && x+r < width && y+r < height) {
    for (int i =  -r; i < r; ++i) {
      xc = x+i;
      for (int j = -r; j < r; ++j) {
        yc = y+i;
        pixel_color = gpuMatElemToFloat(src(yc,xc));
        pixel_color_sharp = gpuMatElemToFloat(src_sharp(yc,xc));

        space_dist2 = (xc-x) * (xc-x) + (yc-y) * (yc-y);

        weight = __expf(space_dist2 * sigma_spatial2_inv_half +
                     norm_l2(pixel_color-center_color) * sigma_color2_inv_half);

        color_sum += weight * pixel_color_sharp;
        norm_sum += weight;
      }
    }
  }
  else {
    for (int i =  -r; i < r; ++i) {
      xc = ::clamp(x+i,0,width-1);

      for (int j = -r; j < r; ++j) {
        yc = ::clamp(y+j,0,height-1);

        pixel_color = gpuMatElemToFloat(src(yc,xc));
        pixel_color_sharp = gpuMatElemToFloat(src_sharp(yc,xc));

        space_dist2 = (xc-x) * (xc-x) + (yc-y) * (yc-y);

        weight = __expf(space_dist2 * sigma_spatial2_inv_half +
                     norm_l2(pixel_color-center_color) * sigma_color2_inv_half);

        color_sum += weight * pixel_color_sharp;
        norm_sum += weight;
      }
    }
  }

   dest(y,x) = floatToGpuMatElem(color_sum/norm_sum);
}

#endif
