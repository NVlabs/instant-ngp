#ifndef BITSTREAM_H
#define BITSTREAM_H

#include <stddef.h>
#include "types.h"

/* forward declaration of opaque type */
typedef struct bitstream bitstream;

extern const size_t stream_word_bits; /* bit stream granularity */

#ifndef _inline
#ifdef __cplusplus
extern "C" {
#endif

/* allocate and initialize bit stream */
bitstream* stream_open(void* buffer, size_t bytes);

/* close and deallocate bit stream */
void stream_close(bitstream* stream);

/* pointer to beginning of stream */
void* stream_data(const bitstream* stream);

/* current byte size of stream (if flushed) */
size_t stream_size(const bitstream* stream);

/* byte capacity of stream */
size_t stream_capacity(const bitstream* stream);

/* number of words per block */
size_t stream_block(const bitstream* stream);

/* number of blocks between consecutive blocks */
int stream_delta(const bitstream* stream);

/* read single bit (0 or 1) */
uint stream_read_bit(bitstream* stream);

/* write single bit */
uint stream_write_bit(bitstream* stream, uint bit);

/* read 0 <= n <= 64 bits */
uint64 stream_read_bits(bitstream* stream, uint n);

/* write 0 <= n <= 64 low bits of value and return remaining bits */
uint64 stream_write_bits(bitstream* stream, uint64 value, uint n);

/* return bit offset to next bit to be read */
size_t stream_rtell(const bitstream* stream);

/* return bit offset to next bit to be written */
size_t stream_wtell(const bitstream* stream);

/* rewind stream to beginning */
void stream_rewind(bitstream* stream);

/* position stream for reading at given bit offset */
void stream_rseek(bitstream* stream, size_t offset);

/* position stream for writing at given bit offset */
void stream_wseek(bitstream* stream, size_t offset);

/* skip over the next n bits */
void stream_skip(bitstream* stream, uint n);

/* append n zero bits to stream */
void stream_pad(bitstream* stream, uint n);

/* align stream on next word boundary */
void stream_align(bitstream* stream);

/* flush out any remaining buffered bits */
void stream_flush(bitstream* stream);

#ifdef BITSTREAM_STRIDED
/* set block size in number of words and spacing in number of blocks */
int stream_set_stride(bitstream* stream, uint block, int delta);
#endif

#ifdef __cplusplus
}
#endif
#endif /* !_inline */

#endif
