static void _t1(inv_lift, Int)(Int* p, uint s);

/* private functions ------------------------------------------------------- */

/* scatter 4*4 block to strided array */
static void
_t2(scatter, Scalar, 2)(const Scalar* q, Scalar* p, int sx, int sy)
{
  uint x, y;
  for (y = 0; y < 4; y++, p += sy - 4 * sx)
    for (x = 0; x < 4; x++, p += sx)
      *p = *q++;
}

/* scatter nx*ny block to strided array */
static void
_t2(scatter_partial, Scalar, 2)(const Scalar* q, Scalar* p, uint nx, uint ny, int sx, int sy)
{
  uint x, y;
  for (y = 0; y < ny; y++, p += sy - nx * sx, q += 4 - nx)
    for (x = 0; x < nx; x++, p += sx, q++)
      *p = *q;
}

/* inverse decorrelating 2D transform */
static void
_t2(inv_xform, Int, 2)(Int* p)
{
  uint x, y;
  /* transform along y */
  for (x = 0; x < 4; x++)
    _t1(inv_lift, Int)(p + 1 * x, 4);
  /* transform along x */
  for (y = 0; y < 4; y++)
    _t1(inv_lift, Int)(p + 4 * y, 1);
}

/* public functions -------------------------------------------------------- */

/* decode 4*4 floating-point block and store at p using strides (sx, sy) */
uint
_t2(zfp_decode_block_strided, Scalar, 2)(zfp_stream* stream, Scalar* p, int sx, int sy)
{
  /* decode contiguous block */
  _cache_align(Scalar fblock[16]);
  uint bits = _t2(zfp_decode_block, Scalar, 2)(stream, fblock);
  /* scatter block to strided array */
  _t2(scatter, Scalar, 2)(fblock, p, sx, sy);
  return bits;
}

/* decode nx*ny floating-point block and store at p using strides (sx, sy) */
uint
_t2(zfp_decode_partial_block_strided, Scalar, 2)(zfp_stream* stream, Scalar* p, uint nx, uint ny, int sx, int sy)
{
  /* decode contiguous block */
  _cache_align(Scalar fblock[16]);
  uint bits = _t2(zfp_decode_block, Scalar, 2)(stream, fblock);
  /* scatter block to strided array */
  _t2(scatter_partial, Scalar, 2)(fblock, p, nx, ny, sx, sy);
  return bits;
}
