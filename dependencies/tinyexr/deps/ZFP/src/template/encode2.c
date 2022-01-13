static void _t1(pad_block, Scalar)(Scalar* p, uint n, uint s);
static void _t1(fwd_lift, Int)(Int* p, uint s);

/* private functions ------------------------------------------------------- */

/* gather 4*4 block from strided array */
static void
_t2(gather, Scalar, 2)(Scalar* q, const Scalar* p, int sx, int sy)
{
  uint x, y;
  for (y = 0; y < 4; y++, p += sy - 4 * sx)
    for (x = 0; x < 4; x++, p += sx)
      *q++ = *p;
}

/* gather nx*ny block from strided array */
static void
_t2(gather_partial, Scalar, 2)(Scalar* q, const Scalar* p, uint nx, uint ny, int sx, int sy)
{
  uint x, y;
  for (y = 0; y < ny; y++, p += sy - nx * sx) {
    for (x = 0; x < nx; x++, p += sx)
      q[4 * y + x] = *p;
    _t1(pad_block, Scalar)(q + 4 * y, nx, 1);
  }
  for (x = 0; x < 4; x++)
    _t1(pad_block, Scalar)(q + x, ny, 4);
}

/* forward decorrelating 2D transform */
static void
_t2(fwd_xform, Int, 2)(Int* p)
{
  uint x, y;
  /* transform along x */
  for (y = 0; y < 4; y++)
    _t1(fwd_lift, Int)(p + 4 * y, 1);
  /* transform along y */
  for (x = 0; x < 4; x++)
    _t1(fwd_lift, Int)(p + 1 * x, 4);
}

/* public functions -------------------------------------------------------- */

/* encode 4*4 floating-point block stored at p using strides (sx, sy) */
uint
_t2(zfp_encode_block_strided, Scalar, 2)(zfp_stream* stream, const Scalar* p, int sx, int sy)
{
  /* gather block from strided array */
  _cache_align(Scalar fblock[16]);
  _t2(gather, Scalar, 2)(fblock, p, sx, sy);
  /* encode floating-point block */
  return _t2(zfp_encode_block, Scalar, 2)(stream, fblock);
}

/* encode nx*ny floating-point block stored at p using strides (sx, sy) */
uint
_t2(zfp_encode_partial_block_strided, Scalar, 2)(zfp_stream* stream, const Scalar* p, uint nx, uint ny, int sx, int sy)
{
  /* gather block from strided array */
  _cache_align(Scalar fblock[16]);
  _t2(gather_partial, Scalar, 2)(fblock, p, nx, ny, sx, sy);
  /* encode floating-point block */
  return _t2(zfp_encode_block, Scalar, 2)(stream, fblock);
}
