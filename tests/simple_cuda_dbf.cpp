#include <iostream>
#include <cstdio>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>

using namespace cv;
using namespace cv::cuda;


void benchmark(const GpuMat& d_src, GpuMat& d_dest, float kernel_size,
               float sigma_color, float sigma_space, int iterations);


int main(int argc, char* argv[]) {
  if ( argc != 2 )
    {
      std::cout << "usage: <binary> <Image_Path>\n";
      return -1;
    }

  Mat src_rgb = imread(argv[1], CV_LOAD_IMAGE_COLOR); 
  if (!src_rgb.data) exit(1);

  Mat src_rgba(src_rgb.rows, src_rgb.cols, CV_8UC4);
  cv::cvtColor(src_rgb, src_rgba, cv::COLOR_RGB2RGBA);


  std::cout << "src rows = " << src_rgb.rows << "\n";
  std::cout << "src cols = " << src_rgb.cols << "\n";

  GpuMat d_src(src_rgba);

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

  GpuMat d_dest(d_src.rows, d_src.cols, d_src.type());

  //TODO: Check bilateral filter OpenCV implementation
  //cuda::bilateralFilter(d_src, d_dest, 41, 20, 150);

  int kernel_size = 30;
  float sigma_color = 0.25;
  float sigma_space = 30.0;
  int iterations = 1;

  benchmark(d_src, d_dest, kernel_size, sigma_color, sigma_space, iterations);

  Mat dest_rgba(d_dest);

  Mat dest_rgb(dest_rgba.rows, dest_rgba.cols, src_rgb.type());
  cv::cvtColor(dest_rgba, dest_rgb, cv::COLOR_RGBA2RGB);

  imwrite("out.jpg", dest_rgb);
  return 0;
}
