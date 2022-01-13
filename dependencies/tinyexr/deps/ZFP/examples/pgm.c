/* simple example that shows how zfp can be used to compress pgm images */

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zfp.h"

int main(int argc, char* argv[])
{
  double rate = 0;
  uint nx, ny;
  uint x, y;
  char line[0x100];
  uchar* image;
  zfp_field* field;
  zfp_stream* zfp;
  bitstream* stream;
  void* buffer;
  size_t bytes;
  size_t size;

  switch (argc) {
    case 2:
      if (sscanf(argv[1], "%lf", &rate) != 1)
        goto usage;
      break;
    default:
    usage:
      fprintf(stderr, "Usage: pgm <rate|-precision> <input.pgm >output.pgm\n");
      return EXIT_FAILURE;
  }

  /* read pgm header */
  if (!fgets(line, sizeof(line), stdin) || strcmp(line, "P5\n") ||
      !fgets(line, sizeof(line), stdin) || sscanf(line, "%u%u", &nx, &ny) != 2 ||
      !fgets(line, sizeof(line), stdin) || strcmp(line, "255\n")) {
    fprintf(stderr, "error opening image\n");
    return EXIT_FAILURE;
  }

  if ((nx & 3u) || (ny & 3u)) {
    fprintf(stderr, "image dimensions must be multiples of four\n");
    return EXIT_FAILURE;
  }

  /* read image data */
  image = malloc(nx * ny);
  if (fread(image, sizeof(*image), nx * ny, stdin) != nx * ny) {
    fprintf(stderr, "error reading image\n");
    return EXIT_FAILURE;
  }

  /* create input array */
  field = zfp_field_2d(image, zfp_type_int32, nx, ny);

  /* initialize compressed stream */
  zfp = zfp_stream_open(NULL);
  if (rate < 0)
    zfp_stream_set_precision(zfp, (uint)floor(0.5 - rate), zfp_type_int32);
  else
    zfp_stream_set_rate(zfp, rate, zfp_type_int32, 2, 0);
  bytes = zfp_stream_maximum_size(zfp, field);
  buffer = malloc(bytes);
  stream = stream_open(buffer, bytes);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_field_free(field);

  /* compress */
  for (y = 0; y < ny; y += 4)
    for (x = 0; x < nx; x += 4) {
      uchar ublock[16];
      int32 iblock[16];
      uint i, j;
      for (j = 0; j < 4; j++)
        for (i = 0; i < 4; i++)
          ublock[i + 4 * j] = image[x + i + nx * (y + j)];
      zfp_promote_uint8_to_int32(iblock, ublock, 2);
      zfp_encode_block_int32_2(zfp, iblock);
    }

  zfp_stream_flush(zfp);
  size = zfp_stream_compressed_size(zfp);
  fprintf(stderr, "%u compressed bytes (%.2f bps)\n", (uint)size, (double)size * CHAR_BIT / (nx * ny));

  /* decompress */
  zfp_stream_rewind(zfp);
  for (y = 0; y < ny; y += 4)
    for (x = 0; x < nx; x += 4) {
      int32 iblock[16];
      uchar ublock[16];
      uint i, j;
      zfp_decode_block_int32_2(zfp, iblock);
      zfp_demote_int32_to_uint8(ublock, iblock, 2);
      for (j = 0; j < 4; j++)
        for (i = 0; i < 4; i++)
          image[x + i + nx * (y + j)] = ublock[i + 4 * j];
    }
  zfp_stream_close(zfp);
  stream_close(stream);
  free(buffer);

  /* output reconstructed image */
  printf("P5\n");
  printf("%u %u\n", nx, ny);
  printf("255\n");
  fwrite(image, sizeof(*image), nx * ny, stdout);
  free(image);

  return 0;
}
