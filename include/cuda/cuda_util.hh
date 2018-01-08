#ifndef _CUDA_UTIL_H_
#define _CUDA_UTIL_H_

#include "cuda_common.hh"

__device__ __forceinline__ float norm_l1(const float& a)  { return ::fabs(a); }
__device__ __forceinline__ float norm_l1(const float2& a) { return ::fabs(a.x) + ::fabs(a.y); }
__device__ __forceinline__ float norm_l1(const float3& a) { return ::fabs(a.x) + ::fabs(a.y) + ::fabs(a.z); }
__device__ __forceinline__ float norm_l1(const float4& a) { return ::fabs(a.x) + ::fabs(a.y) + ::fabs(a.z) + ::fabs(a.w); }

__device__ __forceinline__ float norm_l2(const float2& a) { return ::dot(a,a);}
__device__ __forceinline__ float norm_l2(const float3& a) { return ::dot(a,a);}
__device__ __forceinline__ float norm_l2(const float4& a) { return ::dot(a,a);}

__device__ float4 gpuMatElemToFloat(const uchar4 elem);
__device__ uchar4 floatToGpuMatElem(const float4 val);

int iDivUp(int a, int b);

#endif
