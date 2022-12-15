#include <cstdint>
#include <unistd.h>

#include <tapa.h>
using float_v16 = tapa::vec_t<float, 16>;

template <typename T, typename R>
inline void async_read(tapa::async_mmap<T> & A,
                       tapa::ostream<T> & fifo_A,
                       const R A_len,
                       R & i_req,
                       R & i_resp) {
#pragma HLS inline
  if ((i_req < A_len) &
    !A.read_addr.full()) {
    A.read_addr.try_write(i_req);
    ++i_req;
  }
  if (!fifo_A.full() & !A.read_data.empty()) {
    T tmp;
    A.read_data.try_read(tmp);
    fifo_A.try_write(tmp);
    ++i_resp;
  }
}

void Add(tapa::istream<float_v16>& a,
         tapa::istream<float_v16>& b,
         tapa::ostream<float_v16>& c,
         uint64_t n) {
add_loop:
  for (uint64_t i = 0; i < (n + 15) / 16;) {
#pragma HLS loop_tripcount min=1 max=1024*1024
#pragma HLS pipeline II=1
    float_v16 a_chunk, b_chunk;
    if (!a.empty() && !b.empty()) {
      a.try_read(a_chunk);
      b.try_read(b_chunk);
      c << (a_chunk + b_chunk);
      i++;
    }
  }
}

void Mmap2Stream(tapa::async_mmap<float_v16>& mmap,
                 tapa::ostream<float_v16>& stream,
                 uint64_t n) {
  uint64_t num_iter = (n + 15) / 16;
mmap2stream_loop:
  for (uint64_t i_req = 0, i_resp = 0; i_resp < num_iter;) {
#pragma HLS loop_tripcount min=1 max=1024*1024
#pragma HLS pipeline II=1
    async_read(mmap, stream, num_iter, i_req, i_resp);
  }
}

void Stream2Mmap(tapa::istream<float_v16>& stream,
                 tapa::async_mmap<float_v16>& mmap,
                 uint64_t n) {
  uint64_t num_iter = (n + 15) / 16;
stream2mmap_loop:
  for (uint64_t i_req = 0, i_resp = 0; i_resp < num_iter;) {
#pragma HLS loop_tripcount min=1 max=1024*1024
#pragma HLS pipeline II=1
    if ((i_req < num_iter) &
         !stream.empty() &
         !mmap.write_addr.full() &
         !mmap.write_data.full() ) {
      mmap.write_addr.try_write(i_req);
      float_v16 tmpv16;
      stream.try_read(tmpv16);
      mmap.write_data.try_write(tmpv16);
      ++i_req;
    }
    uint8_t n_resp;
    if (mmap.write_resp.try_read(n_resp)) {
      i_resp += int(n_resp) + 1;
    }
  }
}

void VecAdd(tapa::mmap<float_v16> a,
            tapa::mmap<float_v16> b,
            tapa::mmap<float_v16> c,
            uint64_t n) {
  tapa::stream<float_v16, 2> a_v16("a");
  tapa::stream<float_v16, 2> b_v16("b");
  tapa::stream<float_v16, 2> c_v16("c");

  tapa::task()
      .invoke(Mmap2Stream, a, a_v16, n)
      .invoke(Mmap2Stream, b, b_v16, n)
      .invoke(Add, a_v16, b_v16, c_v16, n)
      .invoke(Stream2Mmap, c_v16, c, n);
}
