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

void Dummy(tapa::istream<float_v16>& in_img_stream, 
           tapa::istream<float_v16>& weight_stream,
           tapa::istream<float_v16>& bias_stream,
           tapa::ostream<float_v16>& out_img_stream, 
           uint64_t kNum, 
           uint64_t kKernel, 
           uint64_t kInImSize,
           uint64_t kOutImSize) {
  float_v16 dummy;
  const uint64_t in_img_aligned_size = (kNum*kInImSize*kInImSize + 15) / 16;
  const uint64_t weight_aligned_size = (kNum*kNum*kKernel*kKernel + 15) / 16;
  const uint64_t bias_aligned_size = (kNum + 15) / 16;
  const uint64_t out_img_aligned_size = (kNum*kOutImSize*kOutImSize + 15) / 16;
  for (uint64_t i = 0; i < in_img_aligned_size; i++) {
    in_img_stream.read();
  }
  for (uint64_t i = 0; i < weight_aligned_size; i++) {
    weight_stream.read();
  }
  for (uint64_t i = 0; i < bias_aligned_size; i++) {
    bias_stream.read();
  }
  for (uint64_t i = 0; i < out_img_aligned_size; i++) {
    out_img_stream << dummy;
  }
}

void Cnn(tapa::mmap<const float_v16> in_img, 
         tapa::mmap<const float_v16> weight, 
         tapa::mmap<const float_v16> bias, 
         tapa::mmap<float_v16> out_img, 
         uint64_t kNum, 
         uint64_t kKernel, 
         uint64_t kInImSize,
         uint64_t kOutImSize) {  
  tapa::stream<float_v16, 2> in_img_stream("in_img");
  tapa::stream<float_v16, 2> weight_stream("weight"); 
  tapa::stream<float_v16, 2> bias_stream("bias"); 
  tapa::stream<float_v16, 2> out_img_stream("out_img");
  uint64_t in_img_size = kNum*kInImSize*kInImSize;
  uint64_t weight_size = kNum*kNum*kKernel*kKernel;
  uint64_t bias_size = kNum;
  uint64_t out_img_size = kNum*kOutImSize*kOutImSize;
  tapa::task()
    .invoke(Mmap2Stream, in_img, in_img_stream, 512)       
    .invoke(Mmap2Stream, weight, weight_stream, 1600)       
    .invoke(Mmap2Stream, bias, bias_stream, 8)
    .invoke(Dummy, in_img_stream, weight_stream, bias_stream, out_img_stream, kNum, kKernel, kInImSize, kOutImSize)
    .invoke(Stream2Mmap, out_img_stream, out_img, 128);
}
