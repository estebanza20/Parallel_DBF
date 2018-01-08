#include "cuda/cuda_transpose.hh"

// Transpose kernel
__global__
void d_transpose(const PtrStep<uchar3> src, PtrStep<uchar3> dest,
                 int width, int height)
{
  __shared__ uchar3 block[BLOCK_DIM][BLOCK_DIM+1];

  // read the matrix tile into shared memory
  unsigned int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
  unsigned int y = blockIdx.y * BLOCK_DIM + threadIdx.y;
   
  if ((x < width) && (y < height))
  {
    block[threadIdx.y][threadIdx.x] = src(y,x);
  }

  __syncthreads();

  // write the transposed matrix tile to global memory
  x = blockIdx.y * BLOCK_DIM + threadIdx.x;
  y = blockIdx.x * BLOCK_DIM + threadIdx.y;

  if ((x < height) && (y < width))
  {
    dest(y,x) = block[threadIdx.x][threadIdx.y];
  }
}


void transpose_GPU(const GpuMat& d_src, GpuMat& d_dest)
{
  int width = d_src.cols;
  int height = d_src.rows;

  d_dest.cols = height;
  d_dest.rows = width;
  d_dest.step = d_dest.cols * d_dest.elemSize();
   
  dim3 grid(iDivUp(width, BLOCK_DIM), iDivUp(height, BLOCK_DIM), 1);
  dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
  d_transpose<<< grid, threads >>>(d_src, d_dest, width, height);
  getLastCudaError("Kernel execution failed");
}
