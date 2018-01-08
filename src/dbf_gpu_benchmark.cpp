#include "dbf_gpu_benchmark.hh"

void dbf_RGB_GPU_simpleBenchmark(const GpuMat& d_src, GpuMat& d_dest,
                                 float kernel_size, float sigma_color,
                                 float sigma_space, int iterations)
{
   StopWatchInterface *timer0 = 0;
   sdkCreateTimer(&timer0);

   // warm-up
   dbf_RGB_GPU_simple(d_src, d_dest, kernel_size, sigma_color, sigma_space);
   checkCudaErrors(cudaDeviceSynchronize());

   sdkStartTimer(&timer0);

   // execute the kernel
   for (int i = 0; i < iterations; i++)
   {
     dbf_RGB_GPU_simple(d_src, d_dest, kernel_size, sigma_color, sigma_space);
   }

   checkCudaErrors(cudaDeviceSynchronize());
   sdkStopTimer(&timer0);

   // check if kernel execution generated an error
   getLastCudaError("Kernel execution failed");

   float total_time = sdkGetTimerValue(&timer0);

   printf("Total Processing time: %f (ms)\n", total_time);
   printf("Mean Processing time: %f (ms)\n", total_time/iterations);
}


void dbf_RGB_GPU_shmemBenchmark(const GpuMat& d_src, GpuMat& d_dest,
                                float kernel_size, float sigma_color,
                                float sigma_space, int iterations)
{
   StopWatchInterface *timer0 = 0;
   sdkCreateTimer(&timer0);

   // warm-up
   dbf_RGB_GPU_shmem(d_src, d_dest, kernel_size, sigma_color, sigma_space);
   checkCudaErrors(cudaDeviceSynchronize());

   sdkStartTimer(&timer0);

   // execute the kernel
   for (int i = 0; i < iterations; i++)
   {
     dbf_RGB_GPU_shmem(d_src, d_dest, kernel_size, sigma_color, sigma_space);
   }

   checkCudaErrors(cudaDeviceSynchronize());
   sdkStopTimer(&timer0);

   // check if kernel execution generated an error
   getLastCudaError("Kernel execution failed");

   float total_time = sdkGetTimerValue(&timer0);

   printf("Total Processing time: %f (ms)\n", total_time);
   printf("Mean Processing time: %f (ms)\n", total_time/iterations);
}


void dbf_RGB_GPU_texBenchmark(const GpuMat& d_src, GpuMat& d_dest,
                              float kernel_size, float sigma_color,
                              float sigma_space, int iterations)
{
   StopWatchInterface *timer0 = 0;
   sdkCreateTimer(&timer0);

   // warm-up
   dbf_RGB_GPU_tex(d_src, d_dest, kernel_size, sigma_color, sigma_space);
   checkCudaErrors(cudaDeviceSynchronize());

   sdkStartTimer(&timer0);

   // execute the kernel
   for (int i = 0; i < iterations; i++)
   {
     dbf_RGB_GPU_tex(d_src, d_dest, kernel_size, sigma_color, sigma_space);
   }

   checkCudaErrors(cudaDeviceSynchronize());
   sdkStopTimer(&timer0);

   // check if kernel execution generated an error
   getLastCudaError("Kernel execution failed");

   float total_time = sdkGetTimerValue(&timer0);

   printf("Total Processing time: %f (ms)\n", total_time);
   printf("Mean Processing time: %f (ms)\n", total_time/iterations);
}
