#define kNum 16
#define kInImSize 8
#define kOutImSize 4
#define kKernel 5
#define kInImgSize (kNum*kInImSize*kInImSize)
#define kWeightSize (kNum*kNum*kKernel*kKernel)
#define kBiasSize (kNum)
#define kOutImgSize (kNum*kOutImSize*kOutImSize)

#define Input(x,y,z)    \
    (in_img_vec[(x)*kInImSize*kInImSize+(y)*kInImSize+(z)])
#define Weight(x,y,z,i) \
    (weight_vec[(x)*kNum*kKernel*kKernel+(y)*kKernel*kKernel+(z)*kKernel+(i)])
#define Bias(x)         \
    (bias_vec[(x)])
#define Output(x,y,z)   \
    (out_img_vec[(x)*kOutImSize*kOutImSize+(y)*kOutImSize+(z)])
#define C(x,y)   \
    (c_vec[x][y])
// #define C(x,y)   \
//     (c_vec[(x)*kImSize+(y)])

using float_v16 = tapa::vec_t<float, 16>;

template <class T>
inline T max(T a, T b) { return a > b ? a : b; }

