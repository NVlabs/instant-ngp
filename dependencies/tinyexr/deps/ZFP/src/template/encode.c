#include <limits.h>
#include <math.h>
#include <stdio.h>
#include "../inline/inline.h"
#include "../inline/bitstream.c"

/* private functions ------------------------------------------------------- */

/* return normalized floating-point exponent for x >= 0 */
static int
_t1(exponent, Scalar)(Scalar x)
{
  if (x > 0) {
    int e;
    FREXP(x, &e);
    /* clamp exponent in case x is denormal */
    return MAX(e, 1 - EBIAS);
  }
  return -EBIAS;
}

/* compute maximum exponent in block of n values */
static int
_t1(exponent_block, Scalar)(const Scalar* p, uint n)
{
  Scalar max = 0;
  do {
    Scalar f = FABS(*p++);
    if (max < f)
      max = f;
  } while (--n);
  return _t1(exponent, Scalar)(max);
}

/* map floating-point number x to integer relative to exponent e */
static Scalar
_t1(quantize, Scalar)(Scalar x, int e)
{
  return LDEXP(x, (CHAR_BIT * (int)sizeof(Scalar) - 2) - e);
}

/* forward block-floating-point transform to signed integers */
static void
_t1(fwd_cast, Scalar)(Int* iblock, const Scalar* fblock, uint n, int emax)
{
  /* compute power-of-two scale factor s */
  Scalar s = _t1(quantize, Scalar)(1, emax);
  /* compute p-bit int y = s*x where x is floating and |y| <= 2^(p-2) - 1 */
  do
    *iblock++ = (Int)(s * *fblock++);
  while (--n);
}

/* pad partial block of width n <= 4 and stride s */
static void
_t1(pad_block, Scalar)(Scalar* p, uint n, uint s)
{
  switch (n) {
    case 0:
      p[0 * s] = 0;
      /* FALLTHROUGH */
    case 1:
      p[1 * s] = p[0 * s];
      /* FALLTHROUGH */
    case 2:
      p[2 * s] = p[1 * s];
      /* FALLTHROUGH */
    case 3:
      p[3 * s] = p[0 * s];
      /* FALLTHROUGH */
    default:
      break;
  }
}

/* forward lifting transform of 4-vector */
static void
_t1(fwd_lift, Int)(Int* p, uint s)
{
  Int x, y, z, w;
  x = *p; p += s;
  y = *p; p += s;
  z = *p; p += s;
  w = *p; p += s;

  /*
  ** non-orthogonal transform
  **        ( 4  4  4  4) (x)
  ** 1/16 * ( 5  1 -1 -5) (y)
  **        (-4  4  4 -4) (z)
  **        (-2  6 -6  2) (w)
  */
  x += w; x >>= 1; w -= x;
  z += y; z >>= 1; y -= z;
  x += z; x >>= 1; z -= x;
  w += y; w >>= 1; y -= w;
  w += y >> 1; y -= w >> 1;

  p -= s; *p = w;
  p -= s; *p = z;
  p -= s; *p = y;
  p -= s; *p = x;
}

/* map two's complement signed integer to negabinary unsigned integer */
static UInt
_t1(int2uint, Int)(Int x)
{
  return ((UInt)x + NBMASK) ^ NBMASK;
}

/* reorder signed coefficients and convert to unsigned integer */
static void
_t1(fwd_order, Int)(UInt* ublock, const Int* iblock, const uchar* perm, uint n)
{
  do
    *ublock++ = _t1(int2uint, Int)(iblock[*perm++]);
  while (--n);
}

/* compress sequence of size unsigned integers */
static uint
_t1(encode_ints, UInt)(bitstream* _restrict stream, uint maxbits, uint maxprec, const UInt* _restrict data, uint size)
{
  /* make a copy of bit stream to avoid aliasing */
  bitstream s = *stream;
  uint intprec = CHAR_BIT * (uint)sizeof(UInt);
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint bits = maxbits;
  uint i, k, m, n;
  uint64 x;

  /* encode one bit plane at a time from MSB to LSB */
  for (k = intprec, n = 0; bits && k-- > kmin;) {
    /* step 1: extract bit plane #k to x */
    x = 0;
    for (i = 0; i < size; i++)
      x += (uint64)((data[i] >> k) & 1u) << i;
    /* step 2: encode first n bits of bit plane */
    m = MIN(n, bits);
    bits -= m;
    x = stream_write_bits(&s, x, m);
    /* step 3: unary run-length encode remainder of bit plane */
    for (; n < size && bits && (bits--, stream_write_bit(&s, !!x)); x >>= 1, n++)
      for (; n < size - 1 && bits && (bits--, !stream_write_bit(&s, x & 1u)); x >>= 1, n++)
        ;
  }

  *stream = s;
  return maxbits - bits;
}

/* encode block of integers */
static uint
_t2(encode_block, Int, DIMS)(bitstream* stream, int minbits, int maxbits, int maxprec, Int* iblock)
{
  int bits;
  _cache_align(UInt ublock[BLOCK_SIZE]);
  /* perform decorrelating transform */
  _t2(fwd_xform, Int, DIMS)(iblock);
  /* reorder signed coefficients and convert to unsigned integer */
  _t1(fwd_order, Int)(ublock, iblock, PERM, BLOCK_SIZE);
  /* encode integer coefficients */
  bits = _t1(encode_ints, UInt)(stream, maxbits, maxprec, ublock, BLOCK_SIZE);
  /* write at least minbits bits by padding with zeros */
  if (bits < minbits) {
    stream_pad(stream, minbits - bits);
    bits = minbits;
  }
  return bits;
}

/* public functions -------------------------------------------------------- */

/* encode contiguous integer block */
uint
_t2(zfp_encode_block, Int, DIMS)(zfp_stream* zfp, const Int* iblock)
{
  _cache_align(Int block[BLOCK_SIZE]);
  uint i;
  /* copy block */
  for (i = 0; i < BLOCK_SIZE; i++)
    block[i] = iblock[i];
  return _t2(encode_block, Int, DIMS)(zfp->stream, zfp->minbits, zfp->maxbits, zfp->maxprec, block);
}

/* encode contiguous floating-point block */
uint
_t2(zfp_encode_block, Scalar, DIMS)(zfp_stream* zfp, const Scalar* fblock)
{
  /* compute maximum exponent */
  int emax = _t1(exponent_block, Scalar)(fblock, BLOCK_SIZE);
  int maxprec = _t2(precision, Scalar, DIMS)(emax, zfp->maxprec, zfp->minexp);
  uint e = maxprec ? emax + EBIAS : 0;
  /* encode block only if biased exponent is nonzero */
  if (e) {
    _cache_align(Int iblock[BLOCK_SIZE]);
    /* encode common exponent; LSB indicates that exponent is nonzero */
    int ebits = EBITS + 1;
    stream_write_bits(zfp->stream, 2 * e + 1, ebits);
    /* perform forward block-floating-point transform */
    _t1(fwd_cast, Scalar)(iblock, fblock, BLOCK_SIZE, emax);
    /* encode integer block */
    return ebits + _t2(encode_block, Int, DIMS)(zfp->stream, zfp->minbits - ebits, zfp->maxbits - ebits, maxprec, iblock);
  }
  else {
    /* write single zero-bit to indicate that all values are zero */
    stream_write_bit(zfp->stream, 0);
    if (zfp->minbits > 1) {
      stream_pad(zfp->stream, zfp->minbits - 1);
      return zfp->minbits;
    }
    else
      return 1;
  }
}
