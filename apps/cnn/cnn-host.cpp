#include <cmath>
#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <tapa.h>
#include "cnn.h"

using std::clog;
using std::endl;
using std::vector;
using std::string;

void Cnn(tapa::mmap<const float_v16> in_img, 
         tapa::mmap<const float_v16> weight, 
         tapa::mmap<const float_v16> bias, 
         tapa::mmap<float_v16> out_img);

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // const uint64_t kNum = 128;
  // const uint64_t kKernel = 5;
  // const uint64_t kInImSize = 8;
  // const uint64_t kOutImSize = 4;

  const uint64_t in_img_aligned_size = ((kInImgSize + 15) / 16) * 16;
  const uint64_t weight_aligned_size = ((kWeightSize + 15) / 16) * 16;
  const uint64_t bias_aligned_size = ((kBiasSize + 15) / 16) * 16;
  const uint64_t out_img_aligned_size = ((kOutImgSize + 15) / 16) * 16;
  vector<float> in_img(in_img_aligned_size, 0.f);
  vector<float> weight(weight_aligned_size, 0.f);
  vector<float> bias(bias_aligned_size, 0.f);
  vector<float> out_img(out_img_aligned_size, 0.f);

  int64_t kernel_time_ns = tapa::invoke(
    Cnn, FLAGS_bitstream, 
    tapa::read_only_mmap<const float>(in_img).reinterpret<const float_v16>(),
    tapa::read_only_mmap<const float>(weight).reinterpret<const float_v16>(), 
    tapa::read_only_mmap<const float>(bias).reinterpret<const float_v16>(), 
    tapa::write_only_mmap<float>(out_img).reinterpret<float_v16>());
  clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << endl;

  return 0;
}
