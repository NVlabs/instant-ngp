static void _t1(pad_block, Scalar)(Scalar* p, uint n, uint s);
static void _t1(fwd_lift, Int)(Int* p, uint s);

/* private functions ------------------------------------------------------- */

/* gather 4-value block from strided array */
static void
_t2(gather, Scalar, 1)(Scalar* q, const Scalar* p, int sx)
{
  uint x;
  for (x = 0; x < 4; x++, p += sx)
    *q++ = *p;
}

/* gather nx-value block from strided array */
static void
_t2(gather_partial, Scalar, 1)(Scalar* q, const Scalar* p, uint nx, int sx)
{
  uint x;
  for (x = 0; x < nx; x++, p += sx)
    q[x] = *p;
  _t1(pad_block, Scalar)(q, nx, 1);
}

/* forward decorrelating 1D transform */
static void
_t2(fwd_xform, Int, 1)(Int* p)
{
  /* transform along x */
  _t1(fwd_lift, Int)(p, 1);
}

/* public functions -------------------------------------------------------- */

/* encode 4-value floating-point block stored at p using stride sx */
uint
_t2(zfp_encode_block_strided, Scalar, 1)(zfp_stream* stream, const Scalar* p, int sx)
{
  /* gather block from strided array */
  _cache_align(Scalar fblock[4]);
  _t2(gather, Scalar, 1)(fblock, p, sx);
  /* encode floating-point block */
  return _t2(zfp_encode_block, Scalar, 1)(stream, fblock);
}

/* encode nx-value floating-point block stored at p using stride sx */
uint
_t2(zfp_encode_partial_block_strided, Scalar, 1)(zfp_stream* stream, const Scalar* p, uint nx, int sx)
{
  /* gather block from strided array */
  _cache_align(Scalar fblock[4]);
  _t2(gather_partial, Scalar, 1)(fblock, p, nx, sx);
  /* encode floating-point block */
  return _t2(zfp_encode_block, Scalar, 1)(stream, fblock);
}
