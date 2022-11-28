#include <cstdint>
#include <unistd.h>

#include <tapa.h>
using float_v16 = tapa::vec_t<float, 16>;

void Add(tapa::istream<float_v16>& a, 
         tapa::istream<float_v16>& b,
         tapa::ostream<float_v16>& c, 
         uint64_t n) {
  float_v16 a_chunk, b_chunk;
  for (uint64_t i = 0; i < (n + 15) / 16;) {
  #pragma HLS loop_tripcount min=1 max=1024*1024  
    if (!a.empty() && !b.empty() && !c.full()) {
      a.try_read(a_chunk);
      b.try_read(b_chunk);
      c.try_write(a_chunk + b_chunk);
      i++;
    }
  }
}

void Mmap2Stream(tapa::mmap<const float_v16> mmap, 
                 uint64_t n,
                 tapa::ostream<float_v16>& stream) {
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

void VecAdd(tapa::mmap<const float_v16> a, 
            tapa::mmap<const float_v16> b,
            tapa::mmap<float_v16> c, 
            uint64_t n) {
  tapa::stream<float_v16, 2> a_v16("a");
  tapa::stream<float_v16, 2> b_v16("b");
  tapa::stream<float_v16, 2> c_v16("c");

  tapa::task()
      .invoke(Mmap2Stream, a, n, a_v16)
      .invoke(Mmap2Stream, b, n, b_v16)
      .invoke(Add, a_v16, b_v16, c_v16, n)
      .invoke(Stream2Mmap, c_v16, c, n);
}
