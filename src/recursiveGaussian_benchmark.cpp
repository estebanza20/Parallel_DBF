/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Recursive Gaussian filter
    sgreen 8/1/08

    This code sample implements a Gaussian blur using Deriche's recursive method:
    http://citeseer.ist.psu.edu/deriche93recursively.html

    This is similar to the box filter sample in the SDK, but it uses the previous
    outputs of the filter as well as the previous inputs. This is also known as an
    IIR (infinite impulse response) filter, since its response to an input impulse
    can last forever.

    The main advantage of this method is that the execution time is independent of
    the filter width.

    The GPU processes columns of the image in parallel. To avoid uncoalesced reads
    for the row pass we transpose the image and then transpose it back again
    afterwards.

    The implementation is based on code from the CImg library:
    http://cimg.sourceforge.net/
    Thanks to David Tschumperl� and all the CImg contributors!
*/

#include "recursiveGaussian.hh"

void benchmark(const GpuMat& d_src, GpuMat& d_dest,
	       float sigma, int order, int iterations)
{
   int nthreads = 64;
   StopWatchInterface *timer0 = 0;
   sdkCreateTimer(&timer0);

   GpuMat d_temp(d_src.rows, d_src.cols, d_src.type());

   // warm-up
   gaussianFilterRGBA(d_src, d_dest, d_temp, sigma, order, nthreads);
   checkCudaErrors(cudaDeviceSynchronize());
   
   sdkStartTimer(&timer0);

   // execute the kernel
   for (int i = 0; i < iterations; i++)
   {
      gaussianFilterRGBA(d_src, d_dest, d_temp, sigma, order, nthreads);
   }

   checkCudaErrors(cudaDeviceSynchronize());
   sdkStopTimer(&timer0);

   // check if kernel execution generated an error
   getLastCudaError("Kernel execution failed");

   float total_time = sdkGetTimerValue(&timer0);

   printf("Total Processing time: %f (ms)\n", total_time);
   printf("Mean Processing time: %f (ms)\n", total_time/iterations);
   printf("%f Mpixels/sec\n", (d_src.cols*d_src.rows*iterations / (total_time / 1000.0f)) / 1e6);

}
