#ifndef _CUDA_UTIL_H_
#define _CUDA_UTIL_H_

#include "cuda_common.hh"

__device__ __forceinline__ float norm_l1(const float& a);
__device__ __forceinline__ float norm_l1(const float2& a);
__device__ __forceinline__ float norm_l1(const float3& a);
__device__ __forceinline__ float norm_l1(const float4& a);

__device__ __forceinline__ float norm_l2(const float2& a);
__device__ __forceinline__ float norm_l2(const float3& a);
__device__ __forceinline__ float norm_l2(const float4& a);

// __device__ float3 gpuMatElemToFloat(const uchar3 elem);
// __device__ uchar3 floatToGpuMatElem(const float3 val);

__device__ float4 gpuMatElemToFloat(const uchar4 elem);
__device__ uchar4 floatToGpuMatElem(const float4 val);

int iDivUp(int a, int b);

#endif
