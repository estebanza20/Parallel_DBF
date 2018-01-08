#include <iostream>
#include <cstdio>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "dbf_gpu_benchmark.hh"

using namespace cv;

int main(int argc, char* argv[]) {
  if ( argc != 2 )
    {
      std::cout << "usage: <binary> <Image_Path>\n";
      return -1;
    }

  // Read input image from indicated location
  Mat src_rgb = imread(argv[1], CV_LOAD_IMAGE_COLOR);
  if (!src_rgb.data) exit(1);

  // Convert input image from RGB to RGBA (better memory alignment)
  // FIXME: Find a way to avoid the need of a color conversion
  Mat src_rgba(src_rgb.rows, src_rgb.cols, CV_8UC4);
  cv::cvtColor(src_rgb, src_rgba, cv::COLOR_RGB2RGBA);

  std::cout << "-------------------------------------------\n";
  std::cout << "input image height = " << src_rgb.rows << "\n";
  std::cout << "input image width = " << src_rgb.cols << "\n";
  std::cout << "-------------------------------------------\n";

  // Copy CPU input image to GPU Mat
  GpuMat d_src(src_rgba);

  std::cout << "-------------------------------------------\n";
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
  std::cout << "-------------------------------------------\n";

  // Define GPU output image with the same size as source
  GpuMat d_dest(d_src.rows, d_src.cols, d_src.type());

  // Deceived Bilateral Filter (DBF) parameters
  int kernel_size = 30;
  float sigma_color = 0.25;
  float sigma_space = 30.0;

  // Benchmark number of iterations
  int iterations = 1;

  // Run DBF Benchmark
  dbf_RGB_GPU_texBenchmark(d_src, d_dest, kernel_size, sigma_color,
                           sigma_space, iterations);

  // Copy GPU output image to CPU Mat
  Mat dest_rgba(d_dest);

  // Convert output image from RGBA to RGB
  Mat dest_rgb(dest_rgba.rows, dest_rgba.cols, src_rgb.type());
  cv::cvtColor(dest_rgba, dest_rgb, cv::COLOR_RGBA2RGB);

  // Write output image to file
  imwrite("out.jpg", dest_rgb);
  return 0;
}
