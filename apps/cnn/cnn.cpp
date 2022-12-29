#include <tapa.h>
#include <unistd.h>
#include <vector>
#include "cnn.h"

void Mmap2Stream(tapa::mmap<const float_v16> mmap,
                 tapa::ostream<float_v16>& stream,
                 uint64_t n) {
  for (uint64_t i = 0; i < (n + 15) / 16; ++i) {
    #pragma HLS loop_tripcount min=1 max=256*228*228
    #pragma HLS pipeline II=1
    stream << mmap[i];
  }
}

void Stream2Mmap(tapa::istream<float_v16>& stream,
                 tapa::mmap<float_v16> mmap,
                 uint64_t n) {
  for (uint64_t i = 0; i < (n + 15) / 16; ++i) {
    #pragma HLS loop_tripcount min=1 max=256*112*112
    #pragma HLS pipeline II=1
    stream >> mmap[i];
  }
}

void Convolution(tapa::istream<float_v16>& in_img_stream,
           tapa::istream<float_v16>& weight_stream,
           tapa::istream<float_v16>& bias_stream,
           tapa::ostream<float_v16>& out_img_stream,
           tapa::ostream<uint64_t>& end_signal) {
  static float C[kImDim][kImDim];
  #pragma HLS array_partition variable=C dim=1 cyclic factor=2
  #pragma HLS array_partition variable=C dim=2 complete
  static float Bias[kNum];
  static float InImg[kNum][kInImDim][kInImDim];
  #pragma HLS array_partition variable=InImg dim=1 cyclic factor=4
  #pragma HLS array_partition variable=InImg dim=3 complete
  static float Weight[kNum][kNum][kKernel][kKernel];
  #pragma HLS array_partition variable=Weight dim=2 cyclic factor=4
  static float OutImg[kNum][kOutImDim][kOutImDim];
  #pragma HLS array_partition variable=OutImg dim=3 complete

  // static float input[kNum][kTileH+kKernel-1][kTileW+kKernel-1];
  // #pragma HLS array_partition variable=input dim=1 cyclic factor=4
  // #pragma HLS array_partition variable=input dim=3 complete
  // static float bias[kNum];
  // static float output[kNum][kTileH/2][kTileW/2];
  // static float C[kTileH][kTileW];
  // static float wt[8];

  read_bias:
  for (uint64_t i = 0; i < (kBiasSize+15)/16; i++) {
    #pragma HLS pipeline II=1
    float_v16 bias_v16 = bias_stream.read();
    for (int pos = 0; pos < 16 && i*16+pos < kBiasSize; ++pos) 
      #pragma HLS pipeline II=1     
      ((float*)Bias)[i*16+pos] = bias_v16[pos];
  }
  read_in_img:
  for (uint64_t i = 0; i < (kInImgSize+15)/16; i++) {
    #pragma HLS pipeline II=1
    float_v16 in_img_v16 = in_img_stream.read();
    for (int pos = 0; pos < 16 && i*16+pos < kInImgSize; ++pos)
      #pragma HLS pipeline II=1
      ((float*)InImg)[i*16+pos] = in_img_v16[pos];
  }
  read_weight:
  for (uint64_t i = 0; i < (kWeightSize+15)/16; i++) {
    #pragma HLS pipeline II=1
    float_v16 weight_v16 = weight_stream.read();
    for (int pos = 0; pos < 16 && i*16+pos < kWeightSize; ++pos)
      #pragma HLS pipeline II=1
      ((float*)Weight)[i*16+pos] = weight_v16[pos];
  }
  
  main_i_loop:
  for (int i = 0; i < kNum; ++i) {

    /* Set Bias */
    float b = Bias[i];
    set_bias_h:
    for (int h = 0; h < kImDim; ++h) {
      #pragma HLS pipeline II=1
      set_bias_w:
      for (int w = 0; w < kImDim; ++w)
        #pragma HLS pipeline II=1
        C[h][w] = b;
    }

    /* Convolution */
    convolution_j:
    for (int j = 0; j < kNum; ++j) {
      convolution_p:
      for (int p = 0; p < kKernel; ++p) {
        convolution_q:
        for (int q = 0; q < kKernel; ++q) {
          float w = Weight[i][j][p][q];
          convolution_h:
          for (int h = 0; h < kImDim; ++h) {
            #pragma HLS pipeline II=1
            convolution_w:
            for (int w = 0; w < kImDim; ++w) {
              #pragma HLS pipeline II=1
              C[h][w] += w * InImg[j][h+p][w+q];
            }
          }
        }
      }
    }

    /* ReLU */
    ReLU_h:
    for (int h = 0; h < kImDim; ++h) {
      #pragma HLS pipeline II=1
      ReLU_w:
      for (int w = 0; w < kImDim; ++w) {
        #pragma HLS pipeline II=1
        C[h][w] = max(0.f, C[h][w]);
      }
    }

    /* Max Pooling */
    max_pooling_h:  
    for (int h = 0; h < kOutImDim; ++h) {
      #pragma HLS pipeline II=1
      max_pooling_w:
      for (int w = 0; w < kOutImDim; ++w) {
        #pragma HLS pipeline II=1
        OutImg[i][h][w] = max(
            max(C[h*2][w*2], C[h*2+1][w*2]),
            max(C[h*2][w*2+1], C[h*2+1][w*2+1]));
      }
    }
  }

  write_out_img:
  for (uint64_t i = 0; i < (kOutImgSize+15)/16; i++) {
    #pragma HLS pipeline II=1
    float_v16 out_img_v16;
    for (int pos = 0; pos < 16 && i*16+pos < kOutImgSize; ++pos)
      #pragma HLS pipeline II=1
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

void Cnn(tapa::mmap<const float_v16> in_img,
         tapa::mmap<const float_v16> weight,
         tapa::mmap<const float_v16> bias,
         tapa::mmap<float_v16> out_img,
         tapa::mmap<uint64_t> cycle_count) {
  tapa::stream<float_v16, 2> in_img_stream("in_img");
  tapa::stream<float_v16, 2> weight_stream("weight");
  tapa::stream<float_v16, 2> bias_stream("bias");
  tapa::stream<float_v16, 2> out_img_stream("out_img");
  tapa::stream<uint64_t, 2> end_signal_stream("end_signal");
  tapa::task()
    .invoke(Timer, end_signal_stream, cycle_count)
    .invoke(Mmap2Stream, in_img, in_img_stream, kInImgSize)
    .invoke(Mmap2Stream, weight, weight_stream, kWeightSize)
    .invoke(Mmap2Stream, bias, bias_stream, kBiasSize)
    .invoke(Convolution, in_img_stream, weight_stream, bias_stream, out_img_stream, end_signal_stream)
    .invoke(Stream2Mmap, out_img_stream, out_img, kOutImgSize);
}
