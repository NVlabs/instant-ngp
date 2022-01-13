// single-precision codec
template <>
struct codec<float> {
  // encode contiguous 1D block
  static void encode_block_1(zfp_stream* zfp, const float* block, uint shape)
  {
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      zfp_encode_partial_block_strided_float_1(zfp, block, nx, 1);
    }
    else
      zfp_encode_block_float_1(zfp, block);
  }

  // encode 1D block from strided storage
  static void encode_block_strided_1(zfp_stream* zfp, const float* p, uint shape, int sx)
  {
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      zfp_encode_partial_block_strided_float_1(zfp, p, nx, sx);
    }
    else
      zfp_encode_block_strided_float_1(zfp, p, sx);
  }

  // encode contiguous 2D block
  static void encode_block_2(zfp_stream* zfp, const float* block, uint shape)
  {
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      zfp_encode_partial_block_strided_float_2(zfp, block, nx, ny, 1, 4);
    }
    else
      zfp_encode_block_float_2(zfp, block);
  }

  // encode 2D block from strided storage
  static void encode_block_strided_2(zfp_stream* zfp, const float* p, uint shape, int sx, int sy)
  {
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      zfp_encode_partial_block_strided_float_2(zfp, p, nx, ny, sx, sy);
    }
    else
      zfp_encode_block_strided_float_2(zfp, p, sx, sy);
  }

  // encode contiguous 3D block
  static void encode_block_3(zfp_stream* zfp, const float* block, uint shape)
  {
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      zfp_encode_partial_block_strided_float_3(zfp, block, nx, ny, nz, 1, 4, 16);
    }
    else
      zfp_encode_block_float_3(zfp, block);
  }

  // encode 3D block from strided storage
  static void encode_block_strided_3(zfp_stream* zfp, const float* p, uint shape, int sx, int sy, int sz)
  {
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      zfp_encode_partial_block_strided_float_3(zfp, p, nx, ny, nz, sx, sy, sz);
    }
    else
      zfp_encode_block_strided_float_3(zfp, p, sx, sy, sz);
  }

  // decode contiguous 1D block
  static void decode_block_1(zfp_stream* zfp, float* block, uint shape)
  {
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      zfp_decode_partial_block_strided_float_1(zfp, block, nx, 1);
    }
    else
      zfp_decode_block_float_1(zfp, block);
  }

  // decode 1D block to strided storage
  static void decode_block_strided_1(zfp_stream* zfp, float* p, uint shape, int sx)
  {
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      zfp_decode_partial_block_strided_float_1(zfp, p, nx, sx);
    }
    else
      zfp_decode_block_strided_float_1(zfp, p, sx);
  }

  // decode contiguous 2D block
  static void decode_block_2(zfp_stream* zfp, float* block, uint shape)
  {
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      zfp_decode_partial_block_strided_float_2(zfp, block, nx, ny, 1, 4);
    }
    else
      zfp_decode_block_float_2(zfp, block);
  }

  // decode 2D block to strided storage
  static void decode_block_strided_2(zfp_stream* zfp, float* p, uint shape, int sx, int sy)
  {
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      zfp_decode_partial_block_strided_float_2(zfp, p, nx, ny, sx, sy);
    }
    else
      zfp_decode_block_strided_float_2(zfp, p, sx, sy);
  }

  // decode contiguous 3D block
  static void decode_block_3(zfp_stream* zfp, float* block, uint shape)
  {
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      zfp_decode_partial_block_strided_float_3(zfp, block, nx, ny, nz, 1, 4, 16);
    }
    else
      zfp_decode_block_float_3(zfp, block);
  }

  // decode 3D block to strided storage
  static void decode_block_strided_3(zfp_stream* zfp, float* p, uint shape, int sx, int sy, int sz)
  {
    if (shape) {
      uint nx = 4 - (shape & 3u); shape >>= 2;
      uint ny = 4 - (shape & 3u); shape >>= 2;
      uint nz = 4 - (shape & 3u); shape >>= 2;
      zfp_decode_partial_block_strided_float_3(zfp, p, nx, ny, nz, sx, sy, sz);
    }
    else
      zfp_decode_block_strided_float_3(zfp, p, sx, sy, sz);
  }

  static const zfp_type type = zfp_type_float;
};
