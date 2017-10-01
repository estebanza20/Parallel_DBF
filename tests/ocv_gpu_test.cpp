#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace cv;
using namespace cv::cuda;


void benchmark(const GpuMat& d_src, GpuMat& d_dest, float sigma, int order, int iterations);


int main(int argc, char* argv[]) {
  if ( argc != 2 )
    {
      std::cout << "usage: <binary> <Image_Path>\n";
      return -1;
    }

  Mat src = imread(argv[1], CV_LOAD_IMAGE_COLOR); 
  if (!src.data) exit(1);
  
  std::cout << "src rows = " << src.rows << "\n";
  std::cout << "src cols = " << src.cols << "\n";

  GpuMat d_src(src);

  std::cout << "\nGpuMat info: \n";
  std::cout << "d_src rows = " << d_src.rows << "\n";
  std::cout << "d_src cols = " << d_src.cols << "\n";
  std::cout << "d_src step = " << d_src.step << "\n";
  std::cout << "d_src channels = " << d_src.channels() << "\n";
  std::cout << "d_src type = " << d_src.type() << "\n";
  std::cout << "d_src depth = " << d_src.depth() << "\n";
  std::cout << "d_src elemSize = " << d_src.elemSize() << "\n";
  std::cout << "d_src elemSize1 = " << d_src.elemSize1() << "\n";
  std::cout << "d_src step1 = " << d_src.step1() << "\n";
  
  GpuMat d_dest;
  d_dest.create(d_src.rows, d_src.cols, d_src.type());

  //TODO: Check bilateral filter OpenCV implementation
  //cuda::bilateralFilter(d_src, d_dest, 41, 20, 150);

  float sigma = 10.0f;
  int order = 0;
  int iterations = 100;
  
  benchmark(d_src, d_dest, sigma, order, iterations);
  
  Mat dest(d_dest);
  imwrite("out.jpg", dest);
  return 0;
}
