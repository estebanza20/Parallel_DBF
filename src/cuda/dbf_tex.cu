#include "cuda/cuda_dbf.hh"

__global__
void d_dbf_RGB_tex(cudaTextureObject_t src, cudaTextureObject_t src_sharp,
                      PtrStepSz<uchar4> dest, const int k_size,
                      const float sigma_spatial2_inv_half,
                      const float sigma_color2_inv_half)
{
  int height = dest.rows;
  int width = dest.cols;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int r = k_size/2;

  float4 pixel_color, pixel_color_sharp;
  float4 center_color = tex2D<float4>(src, x, y);

  float4 color_sum = make_float4(0.0f);
  float norm_sum = 0.0f;

  float weight;
  float space_dist2;

  int xg, yg;

  for (int i = -r; i <= r; ++i) {
    xg = x+i;
    for (int j = -r; j <= r; ++j) {
      yg = y+j;
      pixel_color = tex2D<float4>(src, xg, yg);
      pixel_color_sharp = tex2D<float4>(src_sharp, xg, yg);

      space_dist2 = (xg-x) * (xg-x) + (yg-y) * (yg-y);

      weight = __expf(space_dist2 * sigma_spatial2_inv_half +
                      norm_l2(pixel_color-center_color) * sigma_color2_inv_half);

      color_sum += weight * pixel_color_sharp;
      norm_sum += weight;
    }
  }

  dest(y,x) = floatToGpuMatElem(color_sum/norm_sum);
}


void dbf_RGB_GPU_tex(const GpuMat& d_src, GpuMat& d_dest, int kernel_size,
                     float sigma_color, float sigma_space)
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
  cudaMemcpy2DToArray(cuArraySrc, 0,0, d_src.data, d_src.step, 4*width,
                      height, cudaMemcpyDeviceToDevice);

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
  cudaMemcpy2DToArray(cuArraySrc_sharp, 0,0, d_src.data, d_src.step, 4*width,
                      height, cudaMemcpyDeviceToDevice);

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

  d_dbf_RGB_tex<<< grid, block >>>(texSrc, texSrc_sharp, d_dest, kernel_size,
                                  sigma_spatial2_inv_half,
                                  sigma_color2_inv_half);

  getLastCudaError("Kernel execution failed");
}
