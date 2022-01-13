#include <limits.h>
#include <math.h>
#include "../inline/inline.h"
#include "../inline/bitstream.c"

/* private functions ------------------------------------------------------- */

/* map integer x relative to exponent e to floating-point number */
static Scalar
_t1(dequantize, Scalar)(Int x, int e)
{
  return LDEXP(x, e - (CHAR_BIT * (int)sizeof(Scalar) - 2));
}

/* inverse block-floating-point transform from signed integers */
static void
_t1(inv_cast, Scalar)(const Int* iblock, Scalar* fblock, uint n, int emax)
{
  /* compute power-of-two scale factor s */
  Scalar s = _t1(dequantize, Scalar)(1, emax);
  /* compute p-bit float x = s*y where |y| <= 2^(p-2) - 1 */
  do
    *fblock++ = (Scalar)(s * *iblock++);
  while (--n);
}

/* inverse lifting transform of 4-vector */
static void
_t1(inv_lift, Int)(Int* p, uint s)
{
  Int x, y, z, w;
  x = *p; p += s;
  y = *p; p += s;
  z = *p; p += s;
  w = *p; p += s;

  /*
  ** non-orthogonal transform
  **       ( 4  6 -4 -1) (x)
  ** 1/4 * ( 4  2  4  5) (y)
  **       ( 4 -2  4 -5) (z)
  **       ( 4 -6 -4  1) (w)
  */
  y += w >> 1; w -= y >> 1;
  y += w; w <<= 1; w -= y;
  z += x; x <<= 1; x -= z;
  y += z; z <<= 1; z -= y;
  w += x; x <<= 1; x -= w;

  p -= s; *p = w;
  p -= s; *p = z;
  p -= s; *p = y;
  p -= s; *p = x;
}

/* map two's complement signed integer to negabinary unsigned integer */
static Int
_t1(uint2int, UInt)(UInt x)
{
  return (Int)((x ^ NBMASK) - NBMASK);
}

/* reorder unsigned coefficients and convert to signed integer */
static void
_t1(inv_order, Int)(const UInt* ublock, Int* iblock, const uchar* perm, uint n)
{
  do
    iblock[*perm++] = _t1(uint2int, UInt)(*ublock++);
  while (--n);
}

/* decompress sequence of size unsigned integers */
static uint
_t1(decode_ints, UInt)(bitstream* _restrict stream, uint maxbits, uint maxprec, UInt* _restrict data, uint size)
{
  /* make a copy of bit stream to avoid aliasing */
  bitstream s = *stream;
  uint intprec = CHAR_BIT * (uint)sizeof(UInt);
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint bits = maxbits;
  uint i, k, m, n;
  uint64 x;

  /* initialize data array to all zeros */
  for (i = 0; i < size; i++)
    data[i] = 0;

  /* decode one bit plane at a time from MSB to LSB */
  for (k = intprec, n = 0; bits && k-- > kmin;) {
    /* decode first n bits of bit plane #k */
    m = MIN(n, bits);
    bits -= m;
    x = stream_read_bits(&s, m);
    /* unary run-length decode remainder of bit plane */
    for (; n < size && bits && (bits--, stream_read_bit(&s)); x += (uint64)1 << n++)
      for (; n < size - 1 && bits && (bits--, !stream_read_bit(&s)); n++)
        ;
    /* deposit bit plane from x */
    for (i = 0; x; i++, x >>= 1)
      data[i] += (UInt)(x & 1u) << k;
  }

  *stream = s;
  return maxbits - bits;
}

/* decode block of integers */
uint
_t2(decode_block, Int, DIMS)(bitstream* stream, int minbits, int maxbits, int maxprec, Int* iblock)
{
  int bits;
  _cache_align(UInt ublock[BLOCK_SIZE]);
  /* decode integer coefficients */
  bits = _t1(decode_ints, UInt)(stream, maxbits, maxprec, ublock, BLOCK_SIZE);
  /* read at least minbits bits */
  if (bits < minbits) {
    stream_skip(stream, minbits - bits);
    bits = minbits;
  }
  /* reorder unsigned coefficients and convert to signed integer */
  _t1(inv_order, Int)(ublock, iblock, PERM, BLOCK_SIZE);
  /* perform decorrelating transform */
  _t2(inv_xform, Int, DIMS)(iblock);
  return bits;
}

/* public functions -------------------------------------------------------- */

/* decode contiguous floating-point block */
uint
_t2(zfp_decode_block, Int, DIMS)(zfp_stream* zfp, Int* iblock)
{
  return _t2(decode_block, Int, DIMS)(zfp->stream, zfp->minbits, zfp->maxbits, zfp->maxprec, iblock);
}

/* decode contiguous floating-point block */
uint
_t2(zfp_decode_block, Scalar, DIMS)(zfp_stream* zfp, Scalar* fblock)
{
  /* test if block has nonzero values */
  if (stream_read_bit(zfp->stream)) {
    _cache_align(Int iblock[BLOCK_SIZE]);
    /* decode common exponent */
    uint ebits = EBITS + 1;
    int emax = stream_read_bits(zfp->stream, ebits - 1) - EBIAS;
    int maxprec = _t2(precision, Scalar, DIMS)(emax, zfp->maxprec, zfp->minexp);
    /* decode integer block */
    uint bits = _t2(decode_block, Int, DIMS)(zfp->stream, zfp->minbits - ebits, zfp->maxbits - ebits, maxprec, iblock);
    /* perform inverse block-floating-point transform */
    _t1(inv_cast, Scalar)(iblock, fblock, BLOCK_SIZE, emax);
    return ebits + bits;
  }
  else {
    /* set all values to zero */
    uint i;
    for (i = 0; i < BLOCK_SIZE; i++)
      *fblock++ = 0;
    if (zfp->minbits > 1) {
      stream_skip(zfp->stream, zfp->minbits - 1);
      return zfp->minbits;
    }
    else
      return 1;
  }
}
