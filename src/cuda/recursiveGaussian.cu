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
#include "recursiveGaussian_kernel.cuh"


//Round a / b to nearest higher integer value
int iDivUp(int a, int b)
{
   return (a % b != 0) ? (a / b + 1) : (a / b);
}


void transpose(const GpuMat& d_src, GpuMat& d_dest)
{
   int width = d_src.cols;
   int height = d_src.rows;

   d_dest.cols = height;
   d_dest.rows = width;
   d_dest.step = d_dest.cols * d_dest.elemSize();
   
   dim3 grid(iDivUp(width, BLOCK_DIM), iDivUp(height, BLOCK_DIM), 1);
   dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
   d_transpose<<< grid, threads >>>(d_src, d_dest);
   getLastCudaError("Kernel execution failed");
}


void gaussianFilterRGBA(const GpuMat& d_src, GpuMat& d_dest, GpuMat& d_temp,
			float sigma, int order, int nthreads)
{
   int width = d_src.cols;
   int height = d_src.rows;

   //Make sure GpuMats have the same dimensions
   d_dest.cols = d_temp.cols = width;
   d_dest.rows = d_temp.rows = height;
   d_dest.step = d_temp.step = d_src.step;
   
   //Compute filter coefficients
   const float
      nsigma = sigma < 0.1f ? 0.1f : sigma,
   		       alpha = 1.695f / nsigma,
   		       ema = (float)std::exp(-alpha),
   		       ema2 = (float)std::exp(-2*alpha),
   		       b1 = -2*ema,
   		       b2 = ema2;

   float a0 = 0, a1 = 0, a2 = 0, a3 = 0, coefp = 0, coefn = 0;

   switch (order)
   {
      case 0:
      {
   	 const float k = (1-ema)*(1-ema)/(1+2*alpha*ema-ema2);
   	 a0 = k;
   	 a1 = k*(alpha-1)*ema;
   	 a2 = k*(alpha+1)*ema;
   	 a3 = -k*ema2;
      }
      break;

      case 1:
      {
   	 const float k = (1-ema)*(1-ema)/ema;
   	 a0 = k*ema;
   	 a1 = a3 = 0;
   	 a2 = -a0;
      }
      break;

      case 2:
      {
   	 const float
   	    ea = (float)std::exp(-alpha),
   	    k = -(ema2-1)/(2*alpha*ema),
   	    kn = (-2*(-1+3*ea-3*ea*ea+ea*ea*ea)/(3*ea+1+3*ea*ea+ea*ea*ea));
   	 a0 = kn;
   	 a1 = -kn*(1+k*alpha)*ema;
   	 a2 = kn*(1-k*alpha)*ema;
   	 a3 = -kn*ema2;
      }
      break;

      default:
   	 fprintf(stderr, "gaussianFilter: invalid order parameter!\n");
   	 return;
   }

   coefp = (a0+a1)/(1+b1+b2);
   coefn = (a2+a3)/(1+b1+b2);
   
   d_recursiveGaussian_rgba<<< iDivUp(width, nthreads), nthreads >>>(d_src,
   								     d_temp,
   								     width,
   								     height,
   								     a0, a1, a2, a3,
   								     b1, b2,
   								     coefp,
   								     coefn);
   getLastCudaError("Kernel execution failed");

   transpose(d_temp, d_dest);
   getLastCudaError("transpose: Kernel execution failed");

   //Adjust temp dimensions to dest dimensions for recursive gaussian pass
   d_temp.rows = d_dest.rows;
   d_temp.cols = d_dest.cols;
   d_temp.step = d_dest.step;
   
   d_recursiveGaussian_rgba<<< iDivUp(height, nthreads), nthreads >>>(d_dest,
   								      d_temp,
   								      height,
   								      width,
   								      a0, a1, a2, a3,
   								      b1, b2,
   								      coefp,
   								      coefn);

   getLastCudaError("Kernel execution failed");

   transpose(d_temp, d_dest);
}
