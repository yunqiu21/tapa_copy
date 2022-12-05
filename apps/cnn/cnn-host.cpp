#include <cmath>
#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <tapa.h>

using std::clog;
using std::endl;
using std::vector;
using std::string;
using float_v16 = tapa::vec_t<float, 16>;

void Cnn(tapa::mmap<const float_v16> input, 
         tapa::mmap<const float_v16> weight, 
         tapa::mmap<const float_v16> bias, 
         tapa::mmap<float_v16> output, 
         uint64_t kNum, 
         uint64_t kKernel, 
         uint64_t kInImSize,
         uint64_t kOutImSize);

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const uint64_t kNum = 8;
  const uint64_t kKernel = 5;
  const uint64_t kInImSize = 8;
  const uint64_t kOutImSize = 4;

  const uint64_t input_aligned_size = ((kNum*kInImSize*kInImSize + 15) / 16) * 16;
  const uint64_t weight_aligned_size = ((kNum*kNum*kKernel*kKernel + 15) / 16) * 16;
  const uint64_t bias_aligned_size = ((kNum + 15) / 16) * 16;
  const uint64_t output_aligned_size = ((kNum*kOutImSize*kOutImSize + 15) / 16) * 16;
  vector<float> input(input_aligned_size, 0.f);
  vector<float> weight(weight_aligned_size, 0.f);
  vector<float> bias(bias_aligned_size, 0.f);
  vector<float> output(output_aligned_size, 0.f);

  int64_t kernel_time_ns = tapa::invoke(
    Cnn, FLAGS_bitstream, 
    tapa::read_only_mmap<const float>(input).reinterpret<const float_v16>(),
    tapa::read_only_mmap<const float>(weight).reinterpret<const float_v16>(), 
    tapa::read_only_mmap<const float>(bias).reinterpret<const float_v16>(), 
    tapa::write_only_mmap<float>(output).reinterpret<float_v16>(), 
    kNum, 
    kKernel, 
    kInImSize,
    kOutImSize);
  clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << endl;

  return 0;
}
