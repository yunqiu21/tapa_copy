#include <iostream>
#include <vector>

#include <gflags/gflags.h>
#include <tapa.h>
using float_v16 = tapa::vec_t<float, 16>;

using std::clog;
using std::endl;
using std::vector;

void VecAdd(tapa::mmap<float_v16> a_array, tapa::mmap<float_v16> b_array,
            tapa::mmap<float_v16> c_array, uint64_t n);

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

  const uint64_t n = argc > 1 ? atoll(argv[1]) : 1024 * 1024;
  const uint64_t n_aligned = ((n + 15) / 16) * 16;
  vector<float> a(n_aligned);
  vector<float> b(n_aligned);
  vector<float> c(n_aligned);
  for (uint64_t i = 0; i < n_aligned; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i) * 2;
    c[i] = 0.f;
  }
  int64_t kernel_time_ns = tapa::invoke(
      VecAdd, FLAGS_bitstream, tapa::read_only_mmap<float>(a).reinterpret<float_v16>(),
      tapa::read_only_mmap<float>(b).reinterpret<float_v16>(), tapa::write_only_mmap<float>(c).reinterpret<float_v16>(), n);
  clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << endl;

  uint64_t num_errors = 0;
  const uint64_t threshold = 10;  // only report up to these errors
  for (uint64_t i = 0; i < n; ++i) {
    auto expected = i * 3;
    auto actual = static_cast<uint64_t>(c[i]);
    if (actual != expected) {
      if (num_errors < threshold) {
        clog << "expected: " << expected << ", actual: " << actual << endl;
      } else if (num_errors == threshold) {
        clog << "...";
      }
      ++num_errors;
    }
  }
  if (num_errors == 0) {
    clog << "PASS!" << endl;
  } else {
    if (num_errors > threshold) {
      clog << " (+" << (num_errors - threshold) << " more errors)" << endl;
    }
    clog << "FAIL!" << endl;
  }
  return num_errors > 0 ? 1 : 0;
}
