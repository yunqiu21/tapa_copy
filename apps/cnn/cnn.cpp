#include <tapa.h>
#include <unistd.h>
using float_v16 = tapa::vec_t<float, 16>;

void Mmap2Stream(tapa::mmap<const float_v16> mmap,
                 tapa::ostream<float_v16>& stream,
                 uint64_t n) {
  for (uint64_t i = 0; i < (n + 15) / 16; ++i) {
    stream << mmap[i];
  }
}

void Stream2Mmap(tapa::istream<float_v16>& stream, 
                 tapa::mmap<float_v16> mmap,
                 uint64_t n) {
  for (uint64_t i = 0; i < (n + 15) / 16; ++i) {
    stream >> mmap[i];
  }
}

void Dummy(tapa::istream<float_v16>& input_stream, 
           tapa::istream<float_v16>& weight_stream,
           tapa::istream<float_v16>& bias_stream,
           tapa::ostream<float_v16>& output_stream, 
           uint64_t kNum, 
           uint64_t kKernel, 
           uint64_t kInImSize,
           uint64_t kOutImSize) {
  float_v16 dummy;
  const uint64_t input_aligned_size = (kNum*kInImSize*kInImSize + 15) / 16;
  const uint64_t weight_aligned_size = (kNum*kNum*kKernel*kKernel + 15) / 16;
  const uint64_t bias_aligned_size = (kNum + 15) / 16;
  const uint64_t output_aligned_size = (kNum*kOutImSize*kOutImSize + 15) / 16;
  for (uint64_t i = 0; i < input_aligned_size; i++) {
    input_stream.read();
  }
  for (uint64_t i = 0; i < weight_aligned_size; i++) {
    weight_stream.read();
  }
  for (uint64_t i = 0; i < bias_aligned_size; i++) {
    bias_stream.read();
  }
  for (uint64_t i = 0; i < output_aligned_size; i++) {
    output_stream << dummy;
  }
}

void Cnn(tapa::mmap<const float_v16> input, 
         tapa::mmap<const float_v16> weight, 
         tapa::mmap<const float_v16> bias, 
         tapa::mmap<float_v16> output, 
         uint64_t kNum, 
         uint64_t kKernel, 
         uint64_t kInImSize,
         uint64_t kOutImSize) {  
  tapa::stream<float_v16, 2> input_stream("input");
  tapa::stream<float_v16, 2> weight_stream("weight"); 
  tapa::stream<float_v16, 2> bias_stream("bias"); 
  tapa::stream<float_v16, 2> output_stream("output");
  tapa::task()
      .invoke(Mmap2Stream, "Mmap2Stream", input, input_stream, kNum*kInImSize*kInImSize)       
      .invoke(Mmap2Stream, "Mmap2Stream", weight, weight_stream, kNum*kNum*kKernel*kKernel)       
      .invoke(Mmap2Stream, "Mmap2Stream", bias, bias_stream, kNum)
      .invoke(Dummy, "Dummy", input_stream, weight_stream, bias_stream, output_stream, kNum, kKernel, kInImSize, kOutImSize)
      .invoke(Stream2Mmap, "Stream2Mmap", output_stream, output, kNum*kOutImSize*kOutImSize);
}
