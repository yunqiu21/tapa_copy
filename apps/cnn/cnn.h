#define kNum 16
#define kOutImDim 4
#define kImDim (kOutImDim*2)
#define kKernel 5
#define kInImDim (kImDim+kKernel-1)
#define kInImgSize (kNum*kInImDim*kInImDim)
#define kWeightSize (kNum*kNum*kKernel*kKernel)
#define kBiasSize (kNum)
#define kOutImgSize (kNum*kOutImDim*kOutImDim)

#define Input(x,y,z)    \
    (in_img_vec[(x)*kInImDim*kInImDim+(y)*kInImDim+(z)])
#define Weight(x,y,z,i) \
    (weight_vec[(x)*kNum*kKernel*kKernel+(y)*kKernel*kKernel+(z)*kKernel+(i)])
#define Bias(x)         \
    (bias_vec[(x)])
#define Output(x,y,z)   \
    (out_img_vec[(x)*kOutImDim*kOutImDim+(y)*kOutImDim+(z)])
#define C(x,y)   \
    (c_vec[x][y])
// #define C(x,y)   \
//     (c_vec[(x)*kImDim+(y)])

using float_v16 = tapa::vec_t<float, 16>;

template <class T>
inline T max(T a, T b) { return a > b ? a : b; }

