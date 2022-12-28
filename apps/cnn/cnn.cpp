#include <tapa.h>
#include <unistd.h>
#include <vector>
#include "cnn.h"
using float_v16 = tapa::vec_t<float, 16>;
#define Input(x,y,z)    \
    (in_img_vec[(x)*kInImSize*kInImSize+(y)*kInImSize+(z)])
#define Weight(x,y,z,i) \
    (weight_vec[(x)*kNum*kKernel*kKernel+(y)*kKernel*kKernel+(z)*kKernel+(i)])
#define Bias(x)         \
    (bias_vec[(x)])
#define Output(x,y,z)   \
    (out_img_vec[(x)*kOutImSize*kOutImSize+(y)*kOutImSize+z])
#define C(x,y,z)   \
    (c_vec[(x)*kImSize*kImSize+(y)*kImSize+z])

template <class T>
inline T max(T a, T b) { return a > b ? a : b; }

void Mmap2Stream(tapa::mmap<const float_v16> mmap,
                 tapa::ostream<float_v16>& stream,
                 uint64_t n) {
  for (uint64_t i = 0; i < (n + 15) / 16; ++i) {
#pragma HLS loop_tripcount min=1 max=1024*1024
#pragma HLS pipeline II=1
    printf("Call Mmap2Stream\n");
    stream << mmap[i];
  }
}

void Stream2Mmap(tapa::istream<float_v16>& stream,
                 tapa::mmap<float_v16> mmap,
                 uint64_t n) {
  for (uint64_t i = 0; i < (n + 15) / 16; ++i) {
#pragma HLS loop_tripcount min=1 max=1024*1024
#pragma HLS pipeline II=1
    printf("Call Stream2Mmap\n");
    stream >> mmap[i];
  }
}

void Convolution(tapa::istream<float_v16>& in_img_stream,
           tapa::istream<float_v16>& weight_stream,
           tapa::istream<float_v16>& bias_stream,
           tapa::ostream<float_v16>& out_img_stream) {
  const uint64_t in_img_aligned_size = (kNum*kInImSize*kInImSize + 15) / 16;
  const uint64_t weight_aligned_size = (kNum*kNum*kKernel*kKernel + 15) / 16;
  const uint64_t bias_aligned_size = (kNum + 15) / 16;
  const uint64_t out_img_aligned_size = (kNum*kOutImSize*kOutImSize + 15) / 16;
  const uint64_t kImSize = kOutImSize * 2;
  const uint64_t c_vec_size = kNum*kImSize*kImSize;
  float c_vec[c_vec_size];
  float bias_vec[bias_aligned_size*16];
  float in_img_vec[in_img_aligned_size*16];
  float weight_vec[weight_aligned_size*16];
  float out_img_vec[out_img_aligned_size*16];
  // std::vector<float> bias_vec(bias_aligned_size*16, 0.f);
  // std::vector<float> in_img_vec(in_img_aligned_size*16, 0.f);
  // std::vector<float> weight_vec(weight_aligned_size*16, 0.f);
  // std::vector<float> out_img_vec(out_img_aligned_size*16, 0.f);
  printf("Read data to local\n");
  for (uint64_t i = 0; i < bias_aligned_size; i++) {
    #pragma HLS pipeline II=1
    float_v16 bias_v16 = bias_stream.read();
    for (int pos = 0; pos < 16; ++pos)      
      bias_vec[i*16+pos] = bias_v16[pos];
  }
  for (uint64_t i = 0; i < in_img_aligned_size; i++) {
    #pragma HLS pipeline II=1
    float_v16 in_img_v16 = in_img_stream.read();
    for (int pos = 0; pos < 16; ++pos)
      in_img_vec[i*16+pos] = in_img_v16[pos];
  }
  for (uint64_t i = 0; i < weight_aligned_size; i++) {
    #pragma HLS pipeline II=1
    float_v16 weight_v16 = weight_stream.read();
    for (int pos = 0; pos < 16; ++pos)
      weight_vec[i*16+pos] = weight_v16[pos];
  }
printf("Set bias\n");
set_bias:
  for (int i = 0; i < kNum; ++i) {
    #pragma HLS pipeline II=1
    float b = Bias(i);
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w)
        C(i,h,w) = b;
    }
  }
printf("Convolution\n");
convolution:
  for (int i = 0; i < kNum; ++i) {
    // #pragma HLS pipeline II=1
    for (int j = 0; j < kNum; ++j) {
      for (int p = 0; p < kKernel; ++p) {
        for (int q = 0; q < kKernel; ++q) {
          float w = Weight(i,j,p,q);
          for (int h = 0; h < kImSize; ++h) {
            for (int w = 0; w < kImSize; ++w) {
              C(i,h,w) += w * Input(j,h+p,w+q);
            }
          }
        }
      }
    }
  }
printf("ReLU\n");
ReLU:
  for (int i = 0; i < kNum; ++i) {
    #pragma HLS pipeline II=1
    for (int h = 0; h < kImSize; ++h) {
      for (int w = 0; w < kImSize; ++w) {
        C(i,h,w) = max(0.f, C(i,h,w));
      }
    }
  }
printf("Max pooling\n");
max_pooling:
  for (int i = 0; i < kNum; ++i) {
    #pragma HLS pipeline II=1
    for (int h = 0; h < kOutImSize; ++h) {
      for (int w = 0; w < kOutImSize; ++w) {
        Output(i,h,w) = max(
            max(C(i,h*2,w*2), C(i,h*2+1,w*2)),
            max(C(i,h*2,w*2+1), C(i,h*2+1,w*2+1)));
      }
    }
  }
printf("Write output\n");
output:
  for (uint64_t i = 0; i < out_img_aligned_size; i++) {
    #pragma HLS pipeline II=1
    float_v16 out_img_v16;
    for (int pos = 0; pos < 16; ++pos)
      out_img_v16[pos] = out_img_vec[i*16+pos];
    out_img_stream << out_img_v16;
  }
}

void Cnn(tapa::mmap<const float_v16> in_img,
         tapa::mmap<const float_v16> weight,
         tapa::mmap<const float_v16> bias,
         tapa::mmap<float_v16> out_img) {
  tapa::stream<float_v16, 2> in_img_stream("in_img");
  tapa::stream<float_v16, 2> weight_stream("weight");
  tapa::stream<float_v16, 2> bias_stream("bias");
  tapa::stream<float_v16, 2> out_img_stream("out_img");
  tapa::task()
    .invoke(Mmap2Stream, in_img, in_img_stream, in_img_size)
    .invoke(Mmap2Stream, weight, weight_stream, weight_size)
    .invoke(Mmap2Stream, bias, bias_stream, bias_size)
    .invoke(Convolution, in_img_stream, weight_stream, bias_stream, out_img_stream)
    .invoke(Stream2Mmap, out_img_stream, out_img, out_img_size);
}
