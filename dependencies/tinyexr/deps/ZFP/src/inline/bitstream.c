/*
High-speed in-memory bit stream I/O that supports reading and writing between
0 and 64 bits at a time.  The implementation, which relies heavily on bit
shifts, has been carefully written to ensure that all shifts are between
zero and one less the width of the type being shifted to avoid undefined
behavior.  This occasionally causes somewhat convoluted code.

The following assumptions and restrictions apply:

1. The user must allocate a memory buffer large enough to hold the bit stream,
   whether for reading, writing, or both.  This buffer is associated with the
   bit stream via stream_open(buffer, bytes), which allocates and returns a
   pointer to an opaque bit stream struct.  Call stream_close(stream) to
   deallocate this struct.

2. The stream is either in a read or write state (or, initially, in both
   states).  When done writing, call stream_flush(stream) before entering
   read mode to ensure any buffered bits are output.  To enter read mode,
   call stream_rewind(stream) or stream_rseek(stream, offset) to position
   the stream at the beginning or at a particular bit offset.  Conversely,
   stream_rewind(stream) or stream_wseek(stream, offset) positions the
   stream for writing.  In read mode, the following functions may be called:

     size_t stream_size(stream);
     size_t stream_rtell(stream);
     void stream_rewind(stream);
     void stream_rseek(stream, offset);
     void stream_skip(stream, uint n);
     void stream_align(stream);
     uint stream_read_bit(stream);
     uint64 stream_read_bits(stream, n);

   Each of these read calls has a corresponding write call:

     size_t stream_size(stream);
     size_t stream_wtell(stream);
     void stream_rewind(stream);
     void stream_wseek(stream, offset);
     void stream_pad(stream, n);
     void stream_flush(stream);
     uint stream_write_bit(stream, bit);
     uint64 stream_write_bits(stream, value, n);

3. The stream buffer is an unsigned integer of a user-specified type given
   by the BIT_STREAM_WORD_TYPE macro.  Bits are read and written in units of
   this integer word type.  Supported types are 8, 16, 32, or 64 bits wide.
   The bit width of the buffer is denoted by 'wsize' and can be accessed via
   the global variable stream_word_bits.  A small wsize allows for fine
   granularity reads and writes, and may be preferable when working with many
   small blocks of data that require non-sequential access.  The default
   maximum size of 64 bits ensures maximum speed.  Note that even when
   wsize < 64, it is still possible to read and write up to 64 bits at a time
   using stream_read_bits() and stream_write_bits().

4. If BIT_STREAM_STRIDED is defined, words read from or written to the stream
   may be accessed noncontiguously by setting a power-of-two block size (which
   by default is one word) and a block stride (defaults to zero blocks).  The
   word pointer is always incremented by one word each time a word is accessed.
   Once advanced past a block boundary, the word pointer is also advanced by
   the stride to the next block.  This feature may be used to store blocks of
   data interleaved, e.g. for progressive coding or for noncontiguous parallel
   access to the bit stream  Note that the block size is measured in words,
   while the stride is measured in multiples of the block size.  Strided access
   can have a significant performance penalty.

5. Multiple bits are read and written in order of least to most significant
   bit.  Thus, the statement

       value = stream_write_bits(stream, value, n);

   is essentially equivalent to (but faster than)

       for (i = 0; i < n; i++, value >>= 1)
         stream_write_bit(value & 1);

   when 0 <= n <= 64.  The same holds for read calls, and thus

       value = stream_read_bits(stream, n);

   is essentially equivalent to

       for (i = 0, value = 0; i < n; i++)
         value += (uint64)stream_read_bit() << i;

   Note that it is possible to write fewer bits than the argument 'value'
   holds (possibly even zero bits), in which case any unwritten bits are
   returned.

6. Although the stream_wseek(stream, offset) call allows positioning the
   stream for writing at any bit offset without any data loss (i.e. all
   previously written bits preceding the offset remain valid), for efficiency
   the stream_flush(stream) operation will zero all bits up to the next
   multiple of wsize bits, thus overwriting bits that were previously stored
   at that location.  Consequently, random write access is effectively
   supported only at wsize granularity.  For sequential access, the largest
   possible wsize is preferred due to higher speed.

7. It is up to the user to adhere to these rules.  For performance reasons,
   no error checking is done, and in particular buffer overruns are not
   caught.
*/

#include <limits.h>
#include <stdlib.h>

#ifndef _inline
  #define _inline
#endif

/* bit stream word/buffer type; granularity of stream I/O operations */
#ifdef BIT_STREAM_WORD_TYPE
  /* may be 8-, 16-, 32-, or 64-bit unsigned integer type */
  typedef BIT_STREAM_WORD_TYPE word;
#else
  /* use maximum word size by default for highest speed */
  typedef uint64 word;
#endif

/* number of bits in a buffered word */
#define wsize ((uint)(CHAR_BIT * sizeof(word)))

/* bit stream structure (opaque to caller) */
struct bitstream {
  uint bits;   /* number of buffered bits (0 <= bits < wsize) */
  word buffer; /* buffer for incoming/outgoing bits (buffer < 2^bits) */
  word* ptr;   /* pointer to next word to be read/written */
  word* begin; /* beginning of stream */
  word* end;   /* end of stream (currently unused) */
#ifdef BIT_STREAM_STRIDED
  uint mask;   /* one less the block size in number of words  */
  int delta;   /* number of words between consecutive blocks */
#endif
};

/* private functions ------------------------------------------------------- */

/* read a single word from memory */
static word
stream_read_word(bitstream* s)
{
  word w = *s->ptr++;
#ifdef BIT_STREAM_STRIDED
  if (!((s->ptr - s->begin) & s->mask))
    s->ptr += s->delta;
#endif
  return w;
}

/* write a single word to memory */
static void
stream_write_word(bitstream* s, word value)
{
  *s->ptr++ = value;
#ifdef BIT_STREAM_STRIDED
  if (!((s->ptr - s->begin) & s->mask))
    s->ptr += s->delta;
#endif
}

/* public functions -------------------------------------------------------- */

/* pointer to beginning of stream */
_inline void*
stream_data(const bitstream* s)
{
  return s->begin;
}

/* current byte size of stream (if flushed) */
_inline size_t
stream_size(const bitstream* s)
{
  return sizeof(word) * (s->ptr - s->begin);
}

/* byte capacity of stream */
_inline size_t
stream_capacity(const bitstream* s)
{
  return sizeof(word) * (s->end - s->begin);
}

/* number of blocks between consecutive stream blocks */
_inline int
stream_delta(const bitstream* s)
{
#ifdef BIT_STREAM_STRIDED
  return s->delta / (s->mask + 1);
#else
  return 0;
#endif
}

/* read single bit (0 or 1) */
_inline uint
stream_read_bit(bitstream* s)
{
  uint bit;
  if (!s->bits) {
    s->buffer = stream_read_word(s);
    s->bits = wsize;
  }
  s->bits--;
  bit = (uint)s->buffer & 1u;
  s->buffer >>= 1;
  return bit;
}

/* write single bit (must be 0 or 1) */
_inline uint
stream_write_bit(bitstream* s, uint bit)
{
  s->buffer += (word)bit << s->bits;
  if (++s->bits == wsize) {
    stream_write_word(s, s->buffer);
    s->buffer = 0;
    s->bits = 0;
  }
  return bit;
}

/* read 0 <= n <= 64 bits */
_inline uint64
stream_read_bits(bitstream* s, uint n)
{
  uint64 value = s->buffer;
  if (s->bits < n) {
    /* keep fetching wsize bits until enough bits are buffered */
    do {
      /* assert: 0 <= s->bits < n <= 64 */
      s->buffer = stream_read_word(s);
      value += (uint64)s->buffer << s->bits;
      s->bits += wsize;
    } while (sizeof(s->buffer) < sizeof(value) && s->bits < n);
    /* assert: 1 <= n <= s->bits < n + wsize */
    s->bits -= n;
    if (!s->bits) {
      /* value holds exactly n bits; no need for masking */
      s->buffer = 0;
    }
    else {
      /* assert: 1 <= s->bits < wsize */
      s->buffer >>= wsize - s->bits;
      /* assert: 1 <= n <= 64 */
      value &= ((uint64)2 << (n - 1)) - 1;
    }
  }
  else {
    /* assert: 0 <= n <= s->bits < wsize <= 64 */
    s->bits -= n;
    s->buffer >>= n;
    value &= ((uint64)1 << n) - 1;
  }
  return value;
}

/* write 0 <= n <= 64 low bits of value and return remaining bits */
_inline uint64
stream_write_bits(bitstream* s, uint64 value, uint n)
{
  /* append bit string to buffer */
  s->buffer += value << s->bits;
  s->bits += n;
  /* is buffer full? */
  if (s->bits >= wsize) {
    /* 1 <= n <= 64; decrement n to ensure valid right shifts below */
    value >>= 1;
    n--;
    /* assert: 0 <= n < 64; wsize <= s->bits <= wsize + n */
    do {
      /* output wsize bits while buffer is full */
      s->bits -= wsize;
      /* assert: 0 <= s->bits <= n */
      stream_write_word(s, s->buffer);
      /* assert: 0 <= n - s->bits < 64 */
      s->buffer = value >> (n - s->bits);
    } while (sizeof(s->buffer) < sizeof(value) && s->bits >= wsize);
  }
  /* assert: 0 <= s->bits < wsize */
  s->buffer &= ((word)1 << s->bits) - 1;
  /* assert: 0 <= n < 64 */
  return value >> n;
}

/* return bit offset to next bit to be read */
_inline size_t
stream_rtell(const bitstream* s)
{
  return wsize * (s->ptr - s->begin) - s->bits;
}

/* return bit offset to next bit to be written */
_inline size_t
stream_wtell(const bitstream* s)
{
  return wsize * (s->ptr - s->begin) + s->bits;
}

/* position stream for reading or writing at beginning */
_inline void
stream_rewind(bitstream* s)
{
  s->ptr = s->begin;
  s->buffer = 0;
  s->bits = 0;
}

/* position stream for reading at given bit offset */
_inline void
stream_rseek(bitstream* s, size_t offset)
{
  uint n = offset % wsize;
  s->ptr = s->begin + offset / wsize;
  if (n) {
    s->buffer = stream_read_word(s) >> n;
    s->bits = wsize - n;
  }
  else {
    s->buffer = 0;
    s->bits = 0;
  }
}

/* position stream for writing at given bit offset */
_inline void
stream_wseek(bitstream* s, size_t offset)
{
  uint n = offset % wsize;
  s->ptr = s->begin + offset / wsize;
  if (n) {
    word buffer = *s->ptr;
    buffer &= ((word)1 << n) - 1;
    s->buffer = buffer;
    s->bits = n;
  }
  else {
    s->buffer = 0;
    s->bits = 0;
  }
}

/* skip over the next n bits (n >= 0) */
_inline void
stream_skip(bitstream* s, uint n)
{
  stream_rseek(s, stream_rtell(s) + n);
}

/* append n zero bits to stream (n >= 0) */
_inline void
stream_pad(bitstream* s, uint n)
{
  for (s->bits += n; s->bits >= wsize; s->bits -= wsize) {
    stream_write_word(s, s->buffer);
    s->buffer = 0;
  }
}

/* align stream on next word boundary */
_inline void
stream_align(bitstream* s)
{
  if (s->bits)
    stream_skip(s, s->bits);
}

/* write any remaining buffered bits and align stream on next word boundary */
_inline void
stream_flush(bitstream* s)
{
  if (s->bits)
    stream_pad(s, wsize - s->bits);
}

#ifdef BIT_STREAM_STRIDED
/* set block size in number of words and spacing in number of blocks */
_inline int
stream_set_stride(bitstream* s, uint block, int delta)
{
  /* ensure block size is a power of two */
  if (block & (block - 1))
    return 0;
  s->mask = block - 1;
  s->delta = delta * block;
  return 1;
}
#endif

/* allocate and initialize bit stream to user-allocated buffer */
_inline bitstream*
stream_open(void* buffer, size_t bytes)
{
  bitstream* s = malloc(sizeof(bitstream));
  if (s) {
    s->begin = buffer;
    s->end = s->begin + bytes / sizeof(word);
#ifdef BIT_STREAM_STRIDED
    stream_set_stride(s, 0, 0);
#endif
    stream_rewind(s);
  }
  return s;
}

/* close and deallocate bit stream */
_inline void
stream_close(bitstream* s)
{
  free(s);
}
