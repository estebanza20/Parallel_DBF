
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

  dim3 block (32, 8); //32,8
  dim3 grid (iDivUp(width, block.x), iDivUp(height, block.y));

  float sigma_spatial2_inv_half = -1.0f/(2*sigma_space*sigma_space);
  float sigma_color2_inv_half = -1.0f/(2*sigma_color*sigma_color);

  size_t shared_mem = (block.x + kernel_size) * (block.y + kernel_size) * sizeof(uchar4);

  dim3 kernel_radius_num_blocks (iDivUp(kernel_size/2, block.x), iDivUp(kernel_size/2, block.y));

  std::cout << "shared_mem = " << shared_mem << "\n";
  std::cout << "kernel_radius_num_blocks.x = " << kernel_radius_num_blocks.x << "\n";
  std::cout << "kernel_radius_num_blocks.y = " << kernel_radius_num_blocks.y << "\n";

  d_smemDBF_RGB<<< grid, block, shared_mem >>>(d_src, d_temp, d_dest,
                                               kernel_radius_num_blocks,
                                               kernel_size,
                                               sigma_spatial2_inv_half,
                                               sigma_color2_inv_half);

  getLastCudaError("Kernel execution failed");
}



void texDBF_RGB_GPU(const GpuMat& d_src, GpuMat& d_dest,
                         int kernel_size,
                         float sigma_color,
                         float sigma_space)
{
  StopWatchInterface *timer0 = 0;
  sdkCreateTimer(&timer0);

  int width = d_src.cols;
  int height = d_src.rows;

  sdkStartTimer(&timer0);

  //Specify texture objects parameters

  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeWrap;
  texDesc.addressMode[1] = cudaAddressModeWrap;
  // texDesc.filterMode = cudaFilterModeLinear;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeNormalizedFloat;

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

  //--------- Original source image (Src) ------------

  cudaArray* cuArraySrc;
  cudaMallocArray(&cuArraySrc, &channelDesc, width, height);

  //FIXME: Migrate to native cv::Mat -> CuArray memory copy
  cudaMemcpy2DToArray(cuArraySrc, 0,0, d_src.data, d_src.step, 4*width, height, cudaMemcpyDeviceToDevice);

  //Specify texture for Src
  struct cudaResourceDesc resDescSrc;
  memset(&resDescSrc, 0, sizeof(resDescSrc));
  resDescSrc.resType = cudaResourceTypeArray;
  resDescSrc.res.array.array = cuArraySrc;

  //Create texture object for Src
  cudaTextureObject_t texSrc = 0;
  cudaCreateTextureObject(&texSrc, &resDescSrc, &texDesc, NULL);

  //--------- Source image sharped (Src_sharp) ------------

  // GpuMat d_src_sharp(height, width, d_src.type());
  //FIXME: Implement LoG sharpening
  GpuMat d_src_sharp = d_src;

  //Source image sharped
  cudaArray* cuArraySrc_sharp;
  cudaMallocArray(&cuArraySrc_sharp, &channelDesc, width, height);

  //FIXME: Migrate to native cv::Mat -> CuArray memory copy
  cudaMemcpy2DToArray(cuArraySrc_sharp, 0,0, d_src.data, d_src.step, 4*width, height, cudaMemcpyDeviceToDevice);

  //Specify texture for Src_sharp
  struct cudaResourceDesc resDescSrc_sharp;
  memset(&resDescSrc_sharp, 0, sizeof(resDescSrc_sharp));
  resDescSrc_sharp.resType = cudaResourceTypeArray;
  resDescSrc_sharp.res.array.array = cuArraySrc_sharp;

  //Create texture object for Src_sharp
  cudaTextureObject_t texSrc_sharp = 0;
  cudaCreateTextureObject(&texSrc_sharp, &resDescSrc_sharp, &texDesc, NULL);

  sdkStopTimer(&timer0);

  float total_time = sdkGetTimerValue(&timer0);

  printf("Texture loading time: %f (ms)\n", total_time);

  //------------ Kernel Execution ------------

  float sigma_spatial2_inv_half = -1.0f/(2*sigma_space*sigma_space);
  float sigma_color2_inv_half = -1.0f/(2*sigma_color*sigma_color);

  dim3 block (32, 8);
  dim3 grid (iDivUp(width, block.x), iDivUp(height, block.y));

  d_texDBF_RGB<<< grid, block >>>(texSrc, texSrc_sharp, d_dest,
                                  kernel_size,
                                  sigma_spatial2_inv_half,
                                  sigma_color2_inv_half);

  getLastCudaError("Kernel execution failed");
}
