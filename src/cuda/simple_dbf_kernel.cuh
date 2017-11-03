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

__device__ __forceinline__ float norm_l2(const float2& a) { return ::dot(a,a);}
__device__ __forceinline__ float norm_l2(const float3& a) { return ::dot(a,a);}
__device__ __forceinline__ float norm_l2(const float4& a) { return ::dot(a,a);}


__device__
float4 gpuMatElemToFloat(const uchar4 elem)
{
   float4 r;
   r.x = elem.x/255.0f;
   r.y = elem.y/255.0f;
   r.z = elem.z/255.0f;
   return r;
}

__device__
uchar4 floatToGpuMatElem(const float4 val)
{
   uchar4 r;
   r.x = __saturatef(val.x)*255;
   r.y = __saturatef(val.y)*255;
   r.z = __saturatef(val.z)*255;
   return r;
}


__global__
void d_simpleDBF_RGB(const PtrStepSz<uchar4> src, PtrStep<uchar4> src_sharp, PtrStep<uchar4> dest,
                     const int k_size, const float sigma_spatial2_inv_half, const float sigma_color2_inv_half)
{
  int height = src.rows;
  int width = src.cols;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;


  int r = k_size/2;

  float4 pixel_color, pixel_color_sharp;
  float4 center_color = gpuMatElemToFloat(src(y,x));

  float4 color_sum = make_float4(0.0f);
  float norm_sum = 0.0f;

  float weight;
  float space_dist2;

  int xg, yg;

  if (x-r >=0 && y-r >=0 && x+r < width && y+r < height) {
    for (int i =  -r; i < r; ++i) {
      xg = x+i;
      for (int j = -r; j < r; ++j) {
        yg = y+i;
        pixel_color = gpuMatElemToFloat(src(yg,xg));
        pixel_color_sharp = gpuMatElemToFloat(src_sharp(yg,xg));

        space_dist2 = (xg-x) * (xg-x) + (yg-y) * (yg-y);

        weight = __expf(space_dist2 * sigma_spatial2_inv_half +
                     norm_l2(pixel_color-center_color) * sigma_color2_inv_half);

        color_sum += weight * pixel_color_sharp;
        norm_sum += weight;
      }
    }
  }
  else {
    for (int i =  -r; i < r; ++i) {
      xg = ::clamp(x+i,0,width-1);

      for (int j = -r; j < r; ++j) {
        yg = ::clamp(y+j,0,height-1);

        pixel_color = gpuMatElemToFloat(src(yg,xg));
        pixel_color_sharp = gpuMatElemToFloat(src_sharp(yg,xg));

        space_dist2 = (xg-x) * (xg-x) + (yg-y) * (yg-y);

        weight = __expf(space_dist2 * sigma_spatial2_inv_half +
                     norm_l2(pixel_color-center_color) * sigma_color2_inv_half);

        color_sum += weight * pixel_color_sharp;
        norm_sum += weight;
      }
    }
  }

   dest(y,x) = floatToGpuMatElem(color_sum/norm_sum);
}

__global__
void d_smemDBF_RGB(const PtrStepSz<uchar4> src, PtrStep<uchar4> src_sharp, PtrStep<uchar4> dest,
                   const dim3 r_num_blocks,
                   const int k_size,
                   const float sigma_spatial2_inv_half, const float sigma_color2_inv_half)
{
  int height = src.rows;
  int width = src.cols;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int r = k_size/2;

  extern __shared__ uchar4 image_shmem[];

  int xl, yl, xg, yg;

  int r_num_blocks_x = r_num_blocks.x;
  int r_num_blocks_y = r_num_blocks.y;

  int shmem_size_x = k_size+blockDim.x;
  int shmem_size_y = k_size+blockDim.y;

  for (int bxi = -r_num_blocks_x; bxi <= r_num_blocks_x; ++bxi) {
    xl = threadIdx.x + r + bxi * blockDim.x;
    xg = x + bxi * blockDim.x;

    for (int byi = -r_num_blocks_y; byi <= r_num_blocks_y; ++byi) {
      yl = threadIdx.y + r + byi * blockDim.y;
      yg = y + byi * blockDim.y;

      if (xl >= 0 && yl >= 0 && xl < shmem_size_x && yl < shmem_size_y) {
        xg = ::clamp(xg,0,width-1);
        yg = ::clamp(yg,0,height-1);
        image_shmem[xl + shmem_size_x * yl] = src(yg,xg);
      }
    }
  }

  __syncthreads();


  float4 pixel_color, pixel_color_sharp;
  float4 center_color = gpuMatElemToFloat(src(y,x));

  float4 color_sum = make_float4(0.0f);
  float norm_sum = 0.0f;

  float weight;
  float space_dist2;

  for (int i =  -r; i <= r; ++i) {
    xl = threadIdx.x + r + i;
    xg = x + i;
    for (int j = -r; j <= r; ++j) {
      yl = threadIdx.y + r + j;
      yg = y + j;
      pixel_color = gpuMatElemToFloat(image_shmem[xl + shmem_size_x * yl]);
      //FIXME: Currently using same image_shmem for pixel_color_sharp
      pixel_color_sharp = gpuMatElemToFloat(image_shmem[xl + shmem_size_x * yl]);

      space_dist2 = (xg-x)*(xg-x) + (yg-y)*(yg-y);

      weight = __expf(space_dist2 * sigma_spatial2_inv_half +
                      norm_l2(pixel_color-center_color) * sigma_color2_inv_half);

      color_sum += weight * pixel_color_sharp;
      norm_sum += weight;
    }
  }

   dest(y,x) = floatToGpuMatElem(color_sum/norm_sum);
}


#endif
