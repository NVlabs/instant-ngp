#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "zfp.h"
#include "macros.h"
#include "template/template.h"

#define Scalar float
#include "template/compress.c"
#include "template/decompress.c"
#undef Scalar

#define Scalar double
#include "template/compress.c"
#include "template/decompress.c"
#undef Scalar

/* private functions ------------------------------------------------------- */

static uint
type_precision(zfp_type type)
{
  switch (type) {
    case zfp_type_int32:
      return CHAR_BIT * (uint)sizeof(int32);
    case zfp_type_int64:
      return CHAR_BIT * (uint)sizeof(int64);
    case zfp_type_float:
      return CHAR_BIT * (uint)sizeof(float);
    case zfp_type_double:
      return CHAR_BIT * (uint)sizeof(double);
    default:
      return 0;
  }
}

/* public functions: fields ------------------------------------------------ */

zfp_field*
zfp_field_alloc()
{
  zfp_field* field = malloc(sizeof(zfp_field));
  if (field) {
    field->type = zfp_type_none;
    field->nx = field->ny = field->nz = 0;
    field->sx = field->sy = field->sz = 0;
    field->data = 0;
  }
  return field;
}

zfp_field*
zfp_field_1d(void* data, zfp_type type, uint nx)
{
  zfp_field* field = zfp_field_alloc();
  if (field) {
    field->type = type;
    field->nx = nx;
    field->data = data;
  }
  return field;
}

zfp_field*
zfp_field_2d(void* data, zfp_type type, uint nx, uint ny)
{
  zfp_field* field = zfp_field_alloc();
  if (field) {
    field->type = type;
    field->nx = nx;
    field->ny = ny;
    field->data = data;
  }
  return field;
}

zfp_field*
zfp_field_3d(void* data, zfp_type type, uint nx, uint ny, uint nz)
{
  zfp_field* field = zfp_field_alloc();
  if (field) {
    field->type = type;
    field->nx = nx;
    field->ny = ny;
    field->nz = nz;
    field->data = data;
  }
  return field;
}

void
zfp_field_free(zfp_field* field)
{
  free(field);
}

void*
zfp_field_pointer(const zfp_field* field)
{
  return field->data;
}

zfp_type
zfp_field_type(const zfp_field* field)
{
  return field->type;
}

uint
zfp_field_precision(const zfp_field* field)
{
  return type_precision(field->type);
}

uint
zfp_field_dimensionality(const zfp_field* field)
{
  return field->nx ? field->ny ? field->nz ? 3 : 2 : 1 : 0;
}

size_t
zfp_field_size(const zfp_field* field, uint* size)
{
  if (size)
    switch (zfp_field_dimensionality(field)) {
      case 3:
        size[2] = field->sz;
        /* FALLTHROUGH */
      case 2:
        size[1] = field->sy;
        /* FALLTHROUGH */
      case 1:
        size[0] = field->sx;
        break;
    }
  return (size_t)MAX(field->nx, 1u) * (size_t)MAX(field->ny, 1u) * (size_t)MAX(field->nz, 1u);
}

int
zfp_field_stride(const zfp_field* field, int* stride)
{
  if (stride)
    switch (zfp_field_dimensionality(field)) {
      case 3:
        stride[2] = field->sz ? field->sz : field->nx * field->ny;
        /* FALLTHROUGH */
      case 2:
        stride[1] = field->sy ? field->sy : field->nx;
        /* FALLTHROUGH */
      case 1:
        stride[0] = field->sx ? field->sx : 1;
        break;
    }
  return field->sx || field->sy || field->sz;
}

uint64
zfp_field_metadata(const zfp_field* field)
{
  uint64 meta = 0;
  /* 48 bits for dimensions */
  switch (zfp_field_dimensionality(field)) {
    case 1:
      meta <<= 48; meta += field->nx - 1;
      break;
    case 2:
      meta <<= 24; meta += field->ny - 1;
      meta <<= 24; meta += field->nx - 1;
      break;
    case 3:
      meta <<= 16; meta += field->nz - 1;
      meta <<= 16; meta += field->ny - 1;
      meta <<= 16; meta += field->nx - 1;
      break;
  }
  /* 2 bits for dimensionality (1D, 2D, 3D, 4D) */
  meta <<= 2; meta += zfp_field_dimensionality(field) - 1;
  /* 2 bits for scalar type */
  meta <<= 2; meta += field->type - 1;
  return meta;
}

void
zfp_field_set_pointer(zfp_field* field, void* data)
{
  field->data = data;
}

zfp_type
zfp_field_set_type(zfp_field* field, zfp_type type)
{
  switch (type) {
    case zfp_type_int32:
    case zfp_type_int64:
    case zfp_type_float:
    case zfp_type_double:
      field->type = type;
      return type;
    default:
      return zfp_type_none;
  }
}

void
zfp_field_set_size_1d(zfp_field* field, uint n)
{
  field->nx = n;
  field->ny = 0;
  field->nz = 0;
}

void
zfp_field_set_size_2d(zfp_field* field, uint nx, uint ny)
{
  field->nx = nx;
  field->ny = ny;
  field->nz = 0;
}

void
zfp_field_set_size_3d(zfp_field* field, uint nx, uint ny, uint nz)
{
  field->nx = nx;
  field->ny = ny;
  field->nz = nz;
}

void
zfp_field_set_stride_1d(zfp_field* field, int sx)
{
  field->sx = sx;
  field->sy = 0;
  field->sz = 0;
}

void
zfp_field_set_stride_2d(zfp_field* field, int sx, int sy)
{
  field->sx = sx;
  field->sy = sy;
  field->sz = 0;
}

void
zfp_field_set_stride_3d(zfp_field* field, int sx, int sy, int sz)
{
  field->sx = sx;
  field->sy = sy;
  field->sz = sz;
}

int
zfp_field_set_metadata(zfp_field* field, uint64 meta)
{
  uint dims;
  field->type = (zfp_type)((meta & 0x3u) + 1); meta >>= 2;
  dims = (meta & 0x3u) + 1; meta >>= 2;
  switch (dims) {
    case 1:
      field->nx = (meta & 0xffffffffffffull) + 1; meta >>= 48;
      break;
    case 2:
      field->nx = (meta & 0xffffffull) + 1; meta >>= 24;
      field->ny = (meta & 0xffffffull) + 1; meta >>= 24;
      break;
    case 3:
      field->nx = (meta & 0xffffull) + 1; meta >>= 16;
      field->ny = (meta & 0xffffull) + 1; meta >>= 16;
      field->nz = (meta & 0xffffull) + 1; meta >>= 16;
      break;
  }
  field->sx = field->sy = field->sz = 0;
  return 1;
}

/* public functions: zfp compressed stream --------------------------------- */

zfp_stream*
zfp_stream_open(bitstream* stream)
{
  zfp_stream* zfp = malloc(sizeof(zfp_stream));
  if (zfp) {
    zfp->stream = stream;
    zfp->minbits = ZFP_MIN_BITS;
    zfp->maxbits = ZFP_MAX_BITS;
    zfp->maxprec = ZFP_MAX_PREC;
    zfp->minexp = ZFP_MIN_EXP;
  }
  return zfp;
}

void
zfp_stream_close(zfp_stream* zfp)
{
  free(zfp);
}

bitstream*
zfp_stream_bit_stream(const zfp_stream* zfp)
{
  return zfp->stream;
}

uint64
zfp_stream_mode(const zfp_stream* zfp)
{
  uint64 mode = 0;
  uint minbits;
  uint maxbits;
  uint maxprec;
  uint minexp;

  /* fixed rate? */
  if (zfp->minbits == zfp->maxbits &&
      1 <= zfp->maxbits && zfp->maxbits <= 2048 &&
      zfp->maxprec >= ZFP_MAX_PREC &&
      zfp->minexp <= ZFP_MIN_EXP)
    return zfp->maxbits - 1;

  /* fixed precision? */
  if (zfp->minbits <= ZFP_MIN_BITS &&
      zfp->maxbits >= ZFP_MAX_BITS &&
      1 <= zfp->maxprec && zfp->maxprec <= 128 &&
      zfp->minexp <= ZFP_MIN_EXP)
    return zfp->maxprec + 2047;

  /* fixed accuracy? */
  if (zfp->minbits <= ZFP_MIN_BITS &&
      zfp->maxbits >= ZFP_MAX_BITS &&
      zfp->maxprec >= ZFP_MAX_PREC &&
      -1074 <= zfp->minexp && zfp->minexp <= 843)
    return zfp->minexp + 3251;

  /* encode each parameter separately */
  minbits = MAX(1, MIN(zfp->minbits, 0x8000u)) - 1;
  maxbits = MAX(1, MIN(zfp->maxbits, 0x8000u)) - 1;
  maxprec = MAX(1, MIN(zfp->maxprec, 0x0080u)) - 1;
  minexp = MAX(0, MIN(zfp->minexp + 16495, 0x7fff));
  mode <<= 15; mode += minexp;
  mode <<=  7; mode += maxprec;
  mode <<= 15; mode += maxbits;
  mode <<= 15; mode += minbits;
  mode <<= 12; mode += 0xfffu;

  return mode;
}

void
zfp_stream_params(const zfp_stream* zfp, uint* minbits, uint* maxbits, uint* maxprec, int* minexp)
{
  if (minbits)
    *minbits = zfp->minbits;
  if (maxbits)
    *maxbits = zfp->maxbits;
  if (maxprec)
    *maxprec = zfp->maxprec;
  if (minexp)
    *minexp = zfp->minexp;
}

size_t
zfp_stream_compressed_size(const zfp_stream* zfp)
{
  return stream_size(zfp->stream);
}

size_t
zfp_stream_maximum_size(const zfp_stream* zfp, const zfp_field* field)
{
  uint dims = zfp_field_dimensionality(field);
  uint mx = (MAX(field->nx, 1u) + 3) / 4;
  uint my = (MAX(field->ny, 1u) + 3) / 4;
  uint mz = (MAX(field->nz, 1u) + 3) / 4;
  size_t blocks = (size_t)mx * (size_t)my * (size_t)mz;
  uint values = 1u << (2 * dims);
  uint maxbits = 1;

  if (!dims)
    return 0;
  switch (field->type) {
    case zfp_type_none:
      return 0;
    case zfp_type_float:
      maxbits += 8;
      break;
    case zfp_type_double:
      maxbits += 11;
      break;
    default:
      break;
  }
  maxbits += values - 1 + values * MIN(zfp->maxprec, type_precision(field->type));
  maxbits = MIN(maxbits, zfp->maxbits);
  maxbits = MAX(maxbits, zfp->minbits);
  return ((ZFP_HEADER_BITS + blocks * maxbits + stream_word_bits - 1) & ~(stream_word_bits - 1)) / CHAR_BIT;
}

void
zfp_stream_set_bit_stream(zfp_stream* zfp, bitstream* stream)
{
  zfp->stream = stream;
}

double
zfp_stream_set_rate(zfp_stream* zfp, double rate, zfp_type type, uint dims, int wra)
{
  uint n = 1u << (2 * dims);
  uint bits = (uint)floor(n * rate + 0.5);
  switch (type) {
    case zfp_type_float:
      bits = MAX(bits, 1 + 8u);
      break;
    case zfp_type_double:
      bits = MAX(bits, 1 + 11u);
      break;
    default:
      break;
  }
  if (wra) {
    /* for write random access, round up to next multiple of stream word size */
    bits += stream_word_bits - 1;
    bits &= ~(stream_word_bits - 1);
  }
  zfp->minbits = bits;
  zfp->maxbits = bits;
  zfp->maxprec = type_precision(type);
  zfp->minexp = ZFP_MIN_EXP;
  return (double)bits / n;
}

uint
zfp_stream_set_precision(zfp_stream* zfp, uint precision, zfp_type type)
{
  uint maxprec = type_precision(type);
  zfp->minbits = ZFP_MIN_BITS;
  zfp->maxbits = ZFP_MAX_BITS;
  zfp->maxprec = precision ? MIN(maxprec, precision) : maxprec;
  zfp->minexp = ZFP_MIN_EXP;
  return zfp->maxprec;
}

double
zfp_stream_set_accuracy(zfp_stream* zfp, double tolerance, zfp_type type)
{
  int emin = ZFP_MIN_EXP;
  if (tolerance > 0) {
    /* tolerance = x * 2^emin, with 0.5 <= x < 1 */
    frexp(tolerance, &emin);
    emin--;
    /* assert: 2^emin <= tolerance < 2^(emin+1) */
  }
  zfp->minbits = ZFP_MIN_BITS;
  zfp->maxbits = ZFP_MAX_BITS;
  zfp->maxprec = type_precision(type);
  zfp->minexp = emin;
  return tolerance > 0 ? ldexp(1.0, emin) : 0;
}

int
zfp_stream_set_mode(zfp_stream* zfp, uint64 mode)
{
  if (mode <= ZFP_MODE_SHORT_MAX) {
    /* 12-bit encoding of one of three modes */
    if (mode < 2048) {
      /* fixed rate */
      zfp->minbits = zfp->maxbits = (uint)mode + 1;
      zfp->maxprec = ZFP_MAX_PREC;
      zfp->minexp = ZFP_MIN_EXP;
    }
    else if (mode < 2176) {
      /* fixed precision */
      zfp->minbits = ZFP_MIN_BITS;
      zfp->maxbits = ZFP_MAX_BITS;
      zfp->maxprec = (uint)mode - 2047;
      zfp->minexp = ZFP_MIN_EXP;
    }
    else {
      /* fixed accuracy */
      zfp->minbits = ZFP_MIN_BITS;
      zfp->maxbits = ZFP_MAX_BITS;
      zfp->maxprec = ZFP_MAX_PREC;
      zfp->minexp = (uint)mode - 3251;
    }
  }
  else {
    /* 64-bit encoding */
    mode >>= 12; zfp->minbits = ((uint)mode & 0x7fffu) + 1;
    mode >>= 15; zfp->maxbits = ((uint)mode & 0x7fffu) + 1;
    mode >>= 15; zfp->maxprec = ((uint)mode & 0x007fu) + 1;
    mode >>=  7; zfp->minexp  = ((uint)mode & 0x7fffu) - 16495;
  }

  return 1;
}

int
zfp_stream_set_params(zfp_stream* zfp, uint minbits, uint maxbits, uint maxprec, int minexp)
{
  if (minbits > maxbits || !(0 < maxprec && maxprec <= 64))
    return 0;
  zfp->minbits = minbits;
  zfp->maxbits = maxbits;
  zfp->maxprec = maxprec;
  zfp->minexp = minexp;
  return 1;
}

void
zfp_stream_flush(zfp_stream* zfp)
{
  stream_flush(zfp->stream);
}

void
zfp_stream_align(zfp_stream* zfp)
{
  stream_align(zfp->stream);
}

void
zfp_stream_rewind(zfp_stream* zfp)
{
  stream_rewind(zfp->stream);
}

/* public functions: utility functions --------------------------------------*/

void
zfp_promote_int8_to_int32(int32* oblock, const int8* iblock, uint dims)
{
  uint count = 1u << (2 * dims);
  while (count--)
    *oblock++ = (int32)*iblock++ << 23;
}

void
zfp_promote_uint8_to_int32(int32* oblock, const uint8* iblock, uint dims)
{
  uint count = 1u << (2 * dims);
  while (count--)
    *oblock++ = ((int32)*iblock++ - 0x80) << 23;
}

void
zfp_promote_int16_to_int32(int32* oblock, const int16* iblock, uint dims)
{
  uint count = 1u << (2 * dims);
  while (count--)
    *oblock++ = (int32)*iblock++ << 15;
}

void
zfp_promote_uint16_to_int32(int32* oblock, const uint16* iblock, uint dims)
{
  uint count = 1u << (2 * dims);
  while (count--)
    *oblock++ = ((int32)*iblock++ - 0x8000) << 15;
}

void
zfp_demote_int32_to_int8(int8* oblock, const int32* iblock, uint dims)
{
  uint count = 1u << (2 * dims);
  while (count--) {
    int32 i = *iblock++ >> 23;
    *oblock++ = MAX(-0x80, MIN(i, 0x7f));
  }
}

void
zfp_demote_int32_to_uint8(uint8* oblock, const int32* iblock, uint dims)
{
  uint count = 1u << (2 * dims);
  while (count--) {
    int32 i = (*iblock++ >> 23) + 0x80;
    *oblock++ = MAX(0x00, MIN(i, 0xff));
  }
}

void
zfp_demote_int32_to_int16(int16* oblock, const int32* iblock, uint dims)
{
  uint count = 1u << (2 * dims);
  while (count--) {
    int32 i = *iblock++ >> 15;
    *oblock++ = MAX(-0x8000, MIN(i, 0x7fff));
  }
}

void
zfp_demote_int32_to_uint16(uint16* oblock, const int32* iblock, uint dims)
{
  uint count = 1u << (2 * dims);
  while (count--) {
    int32 i = (*iblock++ >> 15) + 0x8000;
    *oblock++ = MAX(0x0000, MIN(i, 0xffff));
  }
}

/* public functions: compression and decompression --------------------------*/

size_t
zfp_compress(zfp_stream* zfp, const zfp_field* field)
{
  void (*compress[2][3][2])(zfp_stream*, const zfp_field*) = {
    {{ compress_float_1,         compress_double_1 },
     { compress_strided_float_2, compress_strided_double_2 },
     { compress_strided_float_3, compress_strided_double_3 }},
    {{ compress_strided_float_1, compress_strided_double_1 },
     { compress_strided_float_2, compress_strided_double_2 },
     { compress_strided_float_3, compress_strided_double_3 }},
  };
  uint dims = zfp_field_dimensionality(field);
  uint type = field->type;
  uint strided = zfp_field_stride(field, NULL);

  switch (type) {
    case zfp_type_float:
    case zfp_type_double:
      break;
    default:
      return 0;
  }

  compress[strided][dims - 1][type - zfp_type_float](zfp, field);
  stream_flush(zfp->stream);

  return stream_size(zfp->stream);
}

int
zfp_decompress(zfp_stream* zfp, zfp_field* field)
{
  void (*decompress[2][3][2])(zfp_stream*, zfp_field*) = {
    {{ decompress_float_1,         decompress_double_1 },
     { decompress_strided_float_2, decompress_strided_double_2 },
     { decompress_strided_float_3, decompress_strided_double_3 }},
    {{ decompress_strided_float_1, decompress_strided_double_1 },
     { decompress_strided_float_2, decompress_strided_double_2 },
     { decompress_strided_float_3, decompress_strided_double_3 }},
  };
  uint dims = zfp_field_dimensionality(field);
  uint type = field->type;
  uint strided = zfp_field_stride(field, NULL);

  switch (type) {
    case zfp_type_float:
    case zfp_type_double:
      break;
    default:
      return 0;
  }

  decompress[strided][dims - 1][type - zfp_type_float](zfp, field);
  stream_align(zfp->stream);

  return 1;
}

size_t
zfp_write_header(zfp_stream* zfp, const zfp_field* field, uint mask)
{
  size_t bits = 0;
  /* 32-bit magic */
  if (mask & ZFP_HEADER_MAGIC) {
    stream_write_bits(zfp->stream, 'z', 8);
    stream_write_bits(zfp->stream, 'f', 8);
    stream_write_bits(zfp->stream, 'p', 8);
    stream_write_bits(zfp->stream, ZFP_VERSION >> 4, 8);
    bits += ZFP_MAGIC_BITS;
  }
  /* 52-bit field metadata */
  if (mask & ZFP_HEADER_FIELD) {
    uint64 meta = zfp_field_metadata(field);
    stream_write_bits(zfp->stream, meta, ZFP_META_BITS);
    bits += ZFP_META_BITS;
  }
  /* 12- or 64-bit compression parameters */
  if (mask & ZFP_HEADER_PARAMS) {
    uint64 mode = zfp_stream_mode(zfp);
    uint size = mode > ZFP_MODE_SHORT_MAX ? ZFP_MODE_LONG_BITS : ZFP_MODE_SHORT_BITS;
    stream_write_bits(zfp->stream, mode, size);
    bits += size;
  }
  return bits;
}

size_t
zfp_read_header(zfp_stream* zfp, zfp_field* field, uint mask)
{
  size_t bits = 0;
  if (mask & ZFP_HEADER_MAGIC) {
    if (stream_read_bits(zfp->stream, 8) != 'z' ||
        stream_read_bits(zfp->stream, 8) != 'f' ||
        stream_read_bits(zfp->stream, 8) != 'p' ||
        stream_read_bits(zfp->stream, 8) != (ZFP_VERSION >> 4))
      return 0;
    bits += ZFP_MAGIC_BITS;
  }
  if (mask & ZFP_HEADER_FIELD) {
    uint64 meta = stream_read_bits(zfp->stream, ZFP_META_BITS);
    if (!zfp_field_set_metadata(field, meta))
      return 0;
    bits += ZFP_META_BITS;
  }
  if (mask & ZFP_HEADER_PARAMS) {
    uint64 mode = stream_read_bits(zfp->stream, ZFP_MODE_SHORT_BITS);
    bits += ZFP_MODE_SHORT_BITS;
    if (mode > ZFP_MODE_SHORT_MAX) {
      uint size = ZFP_MODE_LONG_BITS - ZFP_MODE_SHORT_BITS;
      mode += stream_read_bits(zfp->stream, size) << ZFP_MODE_SHORT_BITS;
      bits += size;
    }
    if (!zfp_stream_set_mode(zfp, mode))
      return 0;
  }
  return bits;
}
