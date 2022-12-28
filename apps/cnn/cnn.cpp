#include <tapa.h>
#include <unistd.h>
#include <vector>
#include "cnn.h"

void Mmap2Stream(tapa::mmap<const float_v16> mmap,
                 tapa::ostream<float_v16>& stream,
                 uint64_t n) {
  for (uint64_t i = 0; i < (n + 15) / 16; ++i) {
    // #pragma HLS loop_tripcount min=1 max=1024*1024
    #pragma HLS pipeline II=1
    // printf("Call Mmap2Stream\n");
    stream << mmap[i];
  }
  // printf("Here?\n");
}

void Stream2Mmap(tapa::istream<float_v16>& stream,
                 tapa::mmap<float_v16> mmap,
                 uint64_t n) {
  for (uint64_t i = 0; i < (n + 15) / 16; ++i) {
    // #pragma HLS loop_tripcount min=1 max=1024*1024
    #pragma HLS pipeline II=1
    // printf("Call Stream2Mmap\n");
    stream >> mmap[i];
  }
}

void Convolution(tapa::istream<float_v16>& in_img_stream,
           tapa::istream<float_v16>& weight_stream,
           tapa::istream<float_v16>& bias_stream,
           tapa::ostream<float_v16>& out_img_stream,
           tapa::ostream<uint64_t>& end_signal) {
  // printf("Here?\n");
  const uint64_t in_img_v16_size = (kNum*kInImDim*kInImDim + 15) / 16;
  const uint64_t weight_v16_size = (kNum*kNum*kKernel*kKernel + 15) / 16;
  const uint64_t bias_v16_size = (kNum + 15) / 16;
  const uint64_t out_img_v16_size = (kNum*kOutImDim*kOutImDim + 15) / 16;
  static float c_vec[kImDim*kImDim];
  // #pragma HLS array_partition variable=c_vec dim=1 cyclic factor=2
  // #pragma HLS array_partition variable=c_vec dim=2 cyclic factor=2
  static float bias_vec[bias_v16_size*16];
  static float in_img_vec[in_img_v16_size*16];
  // #pragma HLS array_partition variable=in_img_vec dim=1 cyclic factor=4
  static float weight_vec[weight_v16_size*16];
  static float out_img_vec[out_img_v16_size*16];
  read_bias:
  for (uint64_t i = 0; i < bias_v16_size; i++) {
    #pragma HLS pipeline II=1
    float_v16 bias_v16 = bias_stream.read();
    for (int pos = 0; pos < 16; ++pos)      
      bias_vec[i*16+pos] = bias_v16[pos];
  }
  read_in_img:
  for (uint64_t i = 0; i < in_img_v16_size; i++) {
    #pragma HLS pipeline II=1
    float_v16 in_img_v16 = in_img_stream.read();
    for (int pos = 0; pos < 16; ++pos)
      in_img_vec[i*16+pos] = in_img_v16[pos];
  }
  read_weight:
  for (uint64_t i = 0; i < weight_v16_size; i++) {
    #pragma HLS pipeline II=1
    float_v16 weight_v16 = weight_stream.read();
    for (int pos = 0; pos < 16; ++pos)
      weight_vec[i*16+pos] = weight_v16[pos];
  }
  
  main_i_loop:
  for (int i = 0; i < kNum; ++i) {

    // printf("Set bias\n");
    float b = Bias(i);
    set_bias:
    for (int h = 0; h < kImDim; ++h) {
      for (int w = 0; w < kImDim; ++w)
        C(h,w) = b;
    }

    // printf("Convolution\n");
    convolution_j:
    for (int j = 0; j < kNum; ++j) {
      // #pragma HLS pipeline II=1
      convolution_p:
      for (int p = 0; p < kKernel; ++p) {
        convolution_q:
        for (int q = 0; q < kKernel; ++q) {
          float w = Weight(i,j,p,q);
          convolution_h:
          for (int h = 0; h < kImDim; ++h) {
            convolution_w:
            for (int w = 0; w < kImDim; ++w) {
              C(h,w) += w * Input(j,h+p,w+q);
            }
          }
        }
      }
    }

    // printf("ReLU\n");
    ReLU:
    for (int h = 0; h < kImDim; ++h) {
      #pragma HLS pipeline II=1
      for (int w = 0; w < kImDim; ++w) {
        C(h,w) = max(0.f, C(h,w));
      }
    }

    // printf("Max pooling\n");
    max_pooling:  
    for (int h = 0; h < kOutImDim; ++h) {
      #pragma HLS pipeline II=1
      for (int w = 0; w < kOutImDim; ++w) {
        Output(i,h,w) = max(
            max(C(h*2,w*2), C(h*2+1,w*2)),
            max(C(h*2,w*2+1), C(h*2+1,w*2+1)));
      }
    }
  }

  // printf("Write output\n");
  write_out_img:
  for (uint64_t i = 0; i < out_img_v16_size; i++) {
    #pragma HLS pipeline II=1
    float_v16 out_img_v16;
    for (int pos = 0; pos < 16; ++pos)
      out_img_v16[pos] = out_img_vec[i*16+pos];
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
