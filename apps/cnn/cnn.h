#define kNum 16
#define kOutImDim 12
#define kImDim (kOutImDim*2)
#define kKernel 5
#define kInImDim (kImDim+kKernel-1)
#define kInImgSize (kNum*kInImDim*kInImDim)
#define kWeightSize (kNum*kNum*kKernel*kKernel)
#define kBiasSize (kNum)
#define kOutImgSize (kNum*kOutImDim*kOutImDim)

using float_v16 = tapa::vec_t<float, 16>;

template <class T>
inline T max(T a, T b) { return a > b ? a : b; }

#define kTileH (kOutImDim)
#define kTileW (kOutImDim)
#define kTileJ (8)

