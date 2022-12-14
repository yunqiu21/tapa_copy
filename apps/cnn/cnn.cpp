#include <tapa.h>
#include <unistd.h>
#include <vector>
#include "cnn.h"

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

void Mmap2Stream(tapa::async_mmap<float_v16>& mmap,
                 tapa::ostream<float_v16>& stream,
                 uint64_t n) {
  uint64_t num_iter = (n + 15) / 16;
  mmap2stream_loop:
  for (uint64_t i_req = 0, i_resp = 0; i_resp < num_iter;) {
    #pragma HLS loop_tripcount min=1 max=256*228*228
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
    #pragma HLS loop_tripcount min=1 max=256*112*112
    #pragma HLS pipeline II=1
    if ((i_req < num_iter) & !stream.empty()
         & !mmap.write_addr.full() & !mmap.write_data.full() ) {
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

void Convolution(tapa::istream<float_v16>& in_img_stream,
           tapa::istream<float_v16>& weight_stream,
           tapa::istream<float_v16>& bias_stream,
           tapa::ostream<float_v16>& out_img_stream,
           tapa::ostream<uint64_t>& end_signal) {

  /* global variable */
  static float Weight[kNum][kNum][kKernel][kKernel];
  #pragma HLS array_partition variable=Weight dim=2 cyclic factor=4
  #pragma HLS array_partition variable=Weight dim=4 complete
  static float Bias[kNum];
  static float InImg[kNum][kInImDim][kInImDim];
  #pragma HLS array_partition variable=InImg dim=3 complete
  static float OutImg[kNum][kOutImDim][kOutImDim];
  #pragma HLS array_partition variable=OutImg dim=3 complete

  /* tiled */
  static float inimg[kNum][kTileH+kKernel-1][kTileW+kKernel-1];
  #pragma HLS array_partition variable=inimg dim=1 cyclic factor=4
  #pragma HLS array_partition variable=inimg dim=3 complete
  static float outimg[kNum][kTileH/2][kTileW/2];
  #pragma HLS array_partition variable=outimg dim=3 complete
  static float C[kTileH][kTileW];
  #pragma HLS array_partition variable=C dim=2 complete

  float wt[kTileJ];

  read_bias:
  for (uint64_t i = 0; i < (kBiasSize+15)/16; i++) {
    #pragma HLS pipeline II=1
    float_v16 bias_v16 = bias_stream.read();
    for (int pos = 0; pos < 16 && i*16+pos < kBiasSize; ++pos) 
      ((float*)Bias)[i*16+pos] = bias_v16[pos];
  }
  read_in_img:
  for (uint64_t i = 0; i < (kInImgSize+15)/16; i++) {
    #pragma HLS pipeline II=1
    float_v16 in_img_v16 = in_img_stream.read();
    for (int pos = 0; pos < 16 && i*16+pos < kInImgSize; ++pos)
      ((float*)InImg)[i*16+pos] = in_img_v16[pos];
  }
  read_weight:
  for (uint64_t i = 0; i < (kWeightSize+15)/16; i++) {
    #pragma HLS pipeline II=1
    float_v16 weight_v16 = weight_stream.read();
    for (int pos = 0; pos < 16 && i*16+pos < kWeightSize; ++pos)
      ((float*)Weight)[i*16+pos] = weight_v16[pos];
  }
  
  main_loop_tile_h:
  for (int hh = 0; hh < kImDim; hh += kTileH) {

    main_loop_tile_w:
    for (int ww = 0; ww < kImDim; ww += kTileW) {

      read_tile:
      for (int j = 0; j < kNum; ++j) {
        for (int h = 0; h < kTileH+kKernel-1; ++h) {
          #pragma HLS pipeline II=1
          for (int w = 0; w < kTileW+kKernel-1; ++w) {
            inimg[j][h][w] = InImg[j][hh+h][ww+w];
          }
        }
      }

      main_loop_i:
      for (int i = 0; i < kNum; ++i) {

        /* Set Bias */
        float b = Bias[i];
        set_bias_h:
        for (int h = 0; h < kTileH; ++h) {
          #pragma HLS pipeline II=1
          set_bias_w:
          for (int w = 0; w < kTileW; ++w)
            C[h][w] = b;
        }

        /* Convolution */
        convolution_j:
        for (int jj = 0; jj < kNum; jj += kTileJ) {
          convolution_p:
          for (int p = 0; p < kKernel; ++p) {
            convolution_q:
            for (int q = 0; q < kKernel; ++q) {   
              #pragma pipeline II=1           
              for (int j = 0; j < kTileJ; ++j) {
                wt[j] = Weight[i][jj+j][p][q];
              }
              convolution_h:
              for (int h = 0; h < kTileH; ++h) {
                #pragma HLS unroll factor=4
                convolution_w:
                for (int w = 0; w < kTileW; ++w) {
                  float tmp = 0;
                  for (int j = 0; j < kTileJ; ++j) {
                    tmp += wt[j] * inimg[jj+j][h+p][w+q];
                  }
                  C[h][w] += tmp;
                }
              }
            }
          }
        }

        /* ReLU */
        ReLU_h:
        for (int h = 0; h < kTileH; ++h) {
          #pragma HLS pipeline II=1
          ReLU_w:
          for (int w = 0; w < kTileW; ++w) {
            C[h][w] = max(0.f, C[h][w]);
          }
        }

        /* Max Pooling */
        max_pooling_h:  
        for (int h = 0; h < kTileH/2; ++h) {
          #pragma HLS pipeline II=1
          max_pooling_w:
          for (int w = 0; w < kTileW/2; ++w) {
            outimg[i][h][w] = max(
              max(C[h*2][w*2], C[h*2][w*2+1]),
              max(C[h*2+1][w*2], C[h*2+1][w*2+1]));
          }
        }

        write_tile: 
        for (int h = 0; h < kTileH/2; ++h) {
          #pragma HLS pipeline II=1
          for (int w = 0; w < kTileW/2; ++w) {
            OutImg[i][hh+h][ww+w] = outimg[i][h][w];
          }
        }
      }
    }
  }

  write_out_img:
  for (uint64_t i = 0; i < (kOutImgSize+15)/16; i++) {
    #pragma HLS pipeline II=1
    float_v16 out_img_v16;
    for (int pos = 0; pos < 16 && i*16+pos < kOutImgSize; ++pos)
      out_img_v16[pos] = ((float*)OutImg)[i*16+pos];
    out_img_stream << out_img_v16;
  }
  end_signal << 1;
}

void Timer(tapa::istream<uint64_t>& end_signal, tapa::mmap<uint64_t> cycle_count) {
  uint64_t count = 0;
  uint64_t tmp;
  while (!end_signal.try_read(tmp)) {
    count++;
  }
  cycle_count[0] = count;
}

void Cnn(tapa::mmap<float_v16> in_img,
         tapa::mmap<float_v16> weight,
         tapa::mmap<float_v16> bias,
         tapa::mmap<float_v16> out_img,
         tapa::mmap<uint64_t> cycle_count) {
  tapa::stream<float_v16, 8> in_img_stream("in_img");
  tapa::stream<float_v16, 8> weight_stream("weight");
  tapa::stream<float_v16, 8> bias_stream("bias");
  tapa::stream<float_v16, 8> out_img_stream("out_img");
  tapa::stream<uint64_t, 2> end_signal_stream("end_signal");
  tapa::task()
    .invoke(Timer, end_signal_stream, cycle_count)
    .invoke(Mmap2Stream, in_img, in_img_stream, kInImgSize)
    .invoke(Mmap2Stream, weight, weight_stream, kWeightSize)
    .invoke(Mmap2Stream, bias, bias_stream, kBiasSize)
    .invoke(Convolution, in_img_stream, weight_stream, bias_stream, out_img_stream, end_signal_stream)
    .invoke(Stream2Mmap, out_img_stream, out_img, kOutImgSize);
}
