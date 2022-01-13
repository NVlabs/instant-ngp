static void _t1(pad_block, Scalar)(Scalar* p, uint n, uint s);
static void _t1(fwd_lift, Int)(Int* p, uint s);

/* private functions ------------------------------------------------------- */

/* gather 4*4*4 block from strided array */
static void
_t2(gather, Scalar, 3)(Scalar* q, const Scalar* p, int sx, int sy, int sz)
{
  uint x, y, z;
  for (z = 0; z < 4; z++, p += sz - 4 * sy)
    for (y = 0; y < 4; y++, p += sy - 4 * sx)
      for (x = 0; x < 4; x++, p += sx)
        *q++ = *p;
}

/* gather nx*ny*nz block from strided array */
static void
_t2(gather_partial, Scalar, 3)(Scalar* q, const Scalar* p, uint nx, uint ny, uint nz, int sx, int sy, int sz)
{
  uint x, y, z;
  for (z = 0; z < nz; z++, p += sz - ny * sy) {
    for (y = 0; y < ny; y++, p += sy - nx * sx) {
      for (x = 0; x < nx; x++, p += sx)
        q[16 * z + 4 * y + x] = *p; 
      _t1(pad_block, Scalar)(q + 16 * z + 4 * y, nx, 1);
    }
    for (x = 0; x < 4; x++)
      _t1(pad_block, Scalar)(q + 16 * z + x, ny, 4);
  }
  for (y = 0; y < 4; y++)
    for (x = 0; x < 4; x++)
      _t1(pad_block, Scalar)(q + 4 * y + x, nz, 16);
}

/* forward decorrelating 3D transform */
static void
_t2(fwd_xform, Int, 3)(Int* p)
{
  uint x, y, z;
  /* transform along x */
  for (z = 0; z < 4; z++)
    for (y = 0; y < 4; y++)
      _t1(fwd_lift, Int)(p + 4 * y + 16 * z, 1);
  /* transform along y */
  for (x = 0; x < 4; x++)
    for (z = 0; z < 4; z++)
      _t1(fwd_lift, Int)(p + 16 * z + 1 * x, 4);
  /* transform along z */
  for (y = 0; y < 4; y++)
    for (x = 0; x < 4; x++)
      _t1(fwd_lift, Int)(p + 1 * x + 4 * y, 16);
}

/* public functions -------------------------------------------------------- */

/* encode 4*4*4 floating-point block stored at p using strides (sx, sy, sz) */
uint
_t2(zfp_encode_block_strided, Scalar, 3)(zfp_stream* stream, const Scalar* p, int sx, int sy, int sz)
{
  /* gather block from strided array */
  _cache_align(Scalar fblock[64]);
  _t2(gather, Scalar, 3)(fblock, p, sx, sy, sz);
  /* encode floating-point block */
  return _t2(zfp_encode_block, Scalar, 3)(stream, fblock);
}

/* encode nx*ny*nz floating-point block stored at p using strides (sx, sy, sz) */
uint
_t2(zfp_encode_partial_block_strided, Scalar, 3)(zfp_stream* stream, const Scalar* p, uint nx, uint ny, uint nz, int sx, int sy, int sz)
{
  /* gather block from strided array */
  _cache_align(Scalar fblock[64]);
  _t2(gather_partial, Scalar, 3)(fblock, p, nx, ny, nz, sx, sy, sz);
  /* encode floating-point block */
  return _t2(zfp_encode_block, Scalar, 3)(stream, fblock);
}
