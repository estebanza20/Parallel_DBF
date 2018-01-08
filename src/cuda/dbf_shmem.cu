#include "cuda/cuda_dbf.hh"

__global__
void d_dbf_RGB_shmem(const PtrStepSz<uchar4> src, PtrStep<uchar4> src_sharp,
                         PtrStep<uchar4> dest, const dim3 r_num_blocks,
                         const int k_size,
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


void dbf_RGB_GPU_shmem(const GpuMat& d_src, GpuMat& d_dest, int kernel_size,
                       float sigma_color, float sigma_space)
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

  size_t shared_mem = (block.x + kernel_size) * (block.y + kernel_size)
                       * sizeof(uchar4);

  dim3 kernel_radius_num_blocks (iDivUp(kernel_size/2, block.x),
                                 iDivUp(kernel_size/2, block.y));

  std::cout << "shared_mem = " << shared_mem << "\n";
  std::cout << "kernel_radius_num_blocks.x = "
            << kernel_radius_num_blocks.x << "\n";
  std::cout << "kernel_radius_num_blocks.y = "
            << kernel_radius_num_blocks.y << "\n";

  d_dbf_RGB_shmem<<< grid, block, shared_mem >>>(d_src, d_temp, d_dest,
                                               kernel_radius_num_blocks,
                                               kernel_size,
                                               sigma_spatial2_inv_half,
                                               sigma_color2_inv_half);

  getLastCudaError("Kernel execution failed");
}

