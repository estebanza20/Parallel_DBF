#include "cuda/cuda_dbf.hh"

__global__
void d_dbf_RGB_simple(const PtrStepSz<uchar4> src, PtrStep<uchar4> src_sharp,
                      PtrStep<uchar4> dest, const int k_size,
                      const float sigma_spatial2_inv_half,
                      const float sigma_color2_inv_half)
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
    for (int i =  -r; i <= r; ++i) {
      xg = x+i;
      for (int j = -r; j <= r; ++j) {
        yg = y+j;
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
    for (int i =  -r; i <= r; ++i) {
      xg = ::clamp(x+i,0,width-1);

      for (int j = -r; j <= r; ++j) {
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


void dbf_RGB_GPU_simple(const GpuMat& d_src, GpuMat& d_dest, int kernel_size,
                        float sigma_color, float sigma_space)
{
  int width = d_src.cols;
  int height = d_src.rows;

  // GpuMat d_temp(height, width, d_src.type());
  //FIXME: Implement LoG sharpening
  GpuMat d_temp = d_src;

  dim3 block (16, 16); //32,8
  dim3 grid (iDivUp(width, block.x), iDivUp(height, block.y));

  float sigma_spatial2_inv_half = -1.0f/(2*sigma_space*sigma_space);
  float sigma_color2_inv_half = -1.0f/(2*sigma_color*sigma_color);

  d_dbf_RGB_simple<<< grid, block >>>(d_src, d_temp, d_dest,
                                     kernel_size,
                                     sigma_spatial2_inv_half,
                                     sigma_color2_inv_half);

  getLastCudaError("Kernel execution failed");
}
