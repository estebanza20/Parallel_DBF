#include "cuda/cuda_util.hh"

__device__ __forceinline__ float norm_l1(const float& a)  { return ::fabs(a); }
__device__ __forceinline__ float norm_l1(const float2& a) { return ::fabs(a.x) + ::fabs(a.y); }
__device__ __forceinline__ float norm_l1(const float3& a) { return ::fabs(a.x) + ::fabs(a.y) + ::fabs(a.z); }
__device__ __forceinline__ float norm_l1(const float4& a) { return ::fabs(a.x) + ::fabs(a.y) + ::fabs(a.z) + ::fabs(a.w); }

__device__ __forceinline__ float norm_l2(const float2& a) { return ::dot(a,a);}
__device__ __forceinline__ float norm_l2(const float3& a) { return ::dot(a,a);}
__device__ __forceinline__ float norm_l2(const float4& a) { return ::dot(a,a);}


// __device__
// float3 gpuMatElemToFloat(const uchar3 elem)
// {
//   float3 r;
//   r.x = elem.x/255.0f;
//   r.y = elem.y/255.0f;
//   r.z = elem.z/255.0f;
//   return r;
// }


// __device__
// uchar3 floatToGpuMatElem(const float3 val)
// {
//   uchar3 r;
//   r.x = __saturatef(val.x)*255;
//   r.y = __saturatef(val.y)*255;
//   r.z = __saturatef(val.z)*255;
//   return r;
// }


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


//Round a / b to nearest higher integer value
int iDivUp(int a, int b)
{
  return (a % b != 0) ? (a / b + 1) : (a / b);
}
