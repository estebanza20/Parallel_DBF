
#include "simple_dbf.hh"
#include "simple_dbf_kernel.cuh"
#include <iostream>

//Round a / b to nearest higher integer value
int iDivUp(int a, int b)
{
   return (a % b != 0) ? (a / b + 1) : (a / b);
}


void simpleDBF_RGB_GPU(const GpuMat& d_src, GpuMat& d_dest,
                       int kernel_size,
                       float sigma_color,
                       float sigma_space)
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

  d_simpleDBF_RGB<<< grid, block >>>(d_src, d_temp, d_dest,
                                     kernel_size,
                                     sigma_spatial2_inv_half,
                                     sigma_color2_inv_half);

  getLastCudaError("Kernel execution failed");
}

void smemDBF_RGB_GPU(const GpuMat& d_src, GpuMat& d_dest,
                       int kernel_size,
                       float sigma_color,
                       float sigma_space)
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

  size_t shared_mem = (block.x + kernel_size) * (block.y + kernel_size) * sizeof(uchar3);

  dim3 kernel_radius_num_blocks (iDivUp(kernel_size/2, block.x), iDivUp(kernel_size/2, block.y));

  std::cout << "kernel_radius_num_blocks.x = " << kernel_radius_num_blocks.x << "\n";
  std::cout << "kernel_radius_num_blocks.y = " << kernel_radius_num_blocks.y << "\n";

  d_smemDBF_RGB<<< grid, block, shared_mem >>>(d_src, d_temp, d_dest,
                                               kernel_radius_num_blocks,
                                               kernel_size,
                                               sigma_spatial2_inv_half,
                                               sigma_color2_inv_half);

  getLastCudaError("Kernel execution failed");
}

