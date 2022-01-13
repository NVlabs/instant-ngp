/* measure the throughput of encoding and decoding 3D blocks of doubles */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "zfp.h"

/* example 3D block of (reinterpreted) doubles */
static const uint64 block[] = {
0xbf7c3a7bb8495ca9ull,
0xbf79f9d9058ffdafull,
0xbf77c7abd0b61999ull,
0xbf75a42c806bd1daull,
0xbf738f8f740b8ea8ull,
0xbf718a050399fef8ull,
0xbf6f2772ff8c30feull,
0xbf6b59aa63d22f68ull,
0xbf67aaf8b80cff9eull,
0xbf641b9e71983592ull,
0xbf60abd3f723f2b7ull,
0xbf5ab7934169cc04ull,
0xbf54574f6f4897d3ull,
0xbf4c6e39da7fb99bull,
0xbf40ae5826a893d1ull,
0xbf25bce8e19d48e1ull,
0x3f253bfed65904d7ull,
0x3f3f18ab46a04cf3ull,
0x3f4948e7cb74278bull,
0x3f51427b51aeec2eull,
0x3f55a0716d8b4b6bull,
0x3f59be96aeaac56full,
0x3f5d9d3ba7bfd327ull,
0x3f609e608469e93eull,
0x3f624ecbcfa3832cull,
0x3f63e0202ae84b4dull,
0x3f6552a61a3f4812ull,
0x3f66a6ae305af268ull,
0x3f67dc910e9935bcull,
0x3f68f4af65036ff7ull,
0x3f69ef71f24e7182ull,
0x3f6acd4983da7d43ull,
0x3f6b8eaef5b348a0ull,
0x3f6c3423328ffb7aull,
0x3f6cbe2f33d33034ull,
0x3f6d2d64018af3acull,
0x3f6d825ab270c540ull,
0x3f6dbdb46be996ccull,
0x3f6de01a6205cca9ull,
0x3f6dea3dd7813dafull,
0x3f6ddcd81dc33335ull,
0x3f6db8aa94de690full,
0x3f6d7e7eab910d8full,
0x3f6d2f25df44c187ull,
0x3f6ccb79bc0e9844ull,
0x3f6c545bdcaf1795ull,
0x3f6bcab5ea9237c4ull,
0x3f6b2f799dcf639bull,
0x3f6a83a0bd297862ull,
0x3f69c82d1e0ec5deull,
0x3f68fe28a4990e53ull,
0x3f6826a5438d8685ull,
0x3f6742bcfc5cd5b2ull,
0x3f665391df231599ull,
0x3f655a4e0aa7d278ull,
0x3f645823ac5e0b09ull,
0x3f634e4d00643085ull,
0x3f623e0c518426a3ull,
0x3f6128abf933439aull,
0x3f600f7e5f92501cull,
0x3f5de7bbf6db0eb7ull,
0x3f5bae5aa4792e11ull,
0x3f5975adf0453ea2ull,
0x3f57409b1fdc65c4ull,
};

int main(int argc, char* argv[])
{
  size_t n = 0x200000;
  double rate = 1;
  zfp_field* field;
  uint insize;
  zfp_stream* zfp;
  bitstream* stream;
  void* buffer;
  size_t bytes;
  clock_t c;
  double time;
  uint i;

  switch (argc) {
    case 3:
      sscanf(argv[2], "%zu", &n);
      /* FALLTHROUGH */
    case 2:
      sscanf(argv[1], "%lf", &rate);
      break;
  }

  /* declare array to compress */
  field = zfp_field_3d(NULL, zfp_type_double, 4, 4, 4 * n);
  insize = n * sizeof(block);

  /* allocate storage for compressed bit stream */
  zfp = zfp_stream_open(NULL);
  zfp_stream_set_rate(zfp, rate, zfp_field_type(field), zfp_field_dimensionality(field), 0);
  bytes = zfp_stream_maximum_size(zfp, field);
  buffer = malloc(bytes);
  stream = stream_open(buffer, bytes);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_field_free(field);

  /* compress */
  c = clock();
  for (i = 0; i < n; i++)
    zfp_encode_block_double_3(zfp, (const double*)block);
  zfp_stream_flush(zfp);
  time = (double)(clock() - c) / CLOCKS_PER_SEC;
  printf("encode in=%u out=%u %.0f MB/s\n", insize, (uint)stream_size(stream), insize / (1024 * 1024 * time));

  /* decompress */
  zfp_stream_rewind(zfp);
  c = clock();
  for (i = 0; i < n; i++) {
    double a[64];
    zfp_decode_block_double_3(zfp, a);
  }
  time = (double)(clock() - c) / CLOCKS_PER_SEC;
  printf("decode in=%u out=%u %.0f MB/s\n", (uint)stream_size(stream), insize, insize / (1024 * 1024 * time));

  zfp_stream_close(zfp);
  stream_close(stream);
  free(buffer);

  return 0;
}
