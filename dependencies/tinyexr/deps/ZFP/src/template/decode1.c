static void _t1(inv_lift, Int)(Int* p, uint s);

/* private functions ------------------------------------------------------- */

/* scatter 4-value block to strided array */
static void
_t2(scatter, Scalar, 1)(const Scalar* q, Scalar* p, int sx)
{
  uint x;
  for (x = 0; x < 4; x++, p += sx)
    *p = *q++;
}

/* scatter nx-value block to strided array */
static void
_t2(scatter_partial, Scalar, 1)(const Scalar* q, Scalar* p, uint nx, int sx)
{
  uint x;
  for (x = 0; x < nx; x++, p += sx)
   *p = *q++;
}

/* inverse decorrelating 1D transform */
static void
_t2(inv_xform, Int, 1)(Int* p)
{
  /* transform along x */
  _t1(inv_lift, Int)(p, 1);
}

/* public functions -------------------------------------------------------- */

/* decode 4-value floating-point block and store at p using stride sx */
uint
_t2(zfp_decode_block_strided, Scalar, 1)(zfp_stream* stream, Scalar* p, int sx)
{
  /* decode contiguous block */
  _cache_align(Scalar fblock[4]);
  uint bits = _t2(zfp_decode_block, Scalar, 1)(stream, fblock);
  /* scatter block to strided array */
  _t2(scatter, Scalar, 1)(fblock, p, sx);
  return bits;
}

/* decode nx-value floating-point block and store at p using stride sx */
uint
_t2(zfp_decode_partial_block_strided, Scalar, 1)(zfp_stream* stream, Scalar* p, uint nx, int sx)
{
  /* decode contiguous block */
  _cache_align(Scalar fblock[4]);
  uint bits = _t2(zfp_decode_block, Scalar, 1)(stream, fblock);
  /* scatter block to strided array */
  _t2(scatter_partial, Scalar, 1)(fblock, p, nx, sx);
  return bits;
}
