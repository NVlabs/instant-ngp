#ifndef ZFP_CODEC_H
#define ZFP_CODEC_H

#include "zfp.h"

namespace zfp {

// C++ wrappers around libzfp C functions
template <typename Scalar>
struct codec {};

#include "zfpcodecf.h"
#include "zfpcodecd.h"

}

#endif
