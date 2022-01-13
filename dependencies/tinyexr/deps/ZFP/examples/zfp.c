#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "zfp.h"
#include "macros.h"

/*
File I/O is done using the following combinations of i, o, s, and z:
- i   : read uncompressed
- z   : read compressed
- i, s: read uncompressed, print stats
- i, o: read and write uncompressed
- i, z: read uncompressed, write compressed
- z, o: read compressed, write uncompressed

The 7 major tasks to be accomplished are:
- read uncompressed:  i
- read compressed:    !i
- compress:           i
- write compressed:   i && z
- decompress:         o || s || (!i && z)
- write uncompressed: o
- compute stats:      s
*/

/* compute and print reconstruction error */
static void
print_error(const void* fin, const void* fout, zfp_type type, uint n)
{
  const float* ff = fin;
  const double* fd = fin;
  const float* gf = fout;
  const double* gd = fout;
  int single = (type == zfp_type_float);
  double fmin = single ? ff[0] : fd[0];
  double fmax = fmin;
  double erms = 0;
  double ermsn = 0;
  double emax = 0;
  double psnr = 0;
  uint i;

  for (i = 0; i < n; i++) {
    double d = fabs(single ? ff[i] - gf[i] : fd[i] - gd[i]);
    double val = single ? ff[i] : fd[i];
    emax = MAX(emax, d);
    erms += d * d;
    fmin = MIN(fmin, val);
    fmax = MAX(fmax, val);
  }
  erms = sqrt(erms / n);
  ermsn = erms / (fmax - fmin);
  psnr = 20 * log10((fmax - fmin) / (2 * erms));
  fprintf(stderr, " rmse=%.4g nrmse=%.4g maxe=%.4g psnr=%.2f", erms, ermsn, emax, psnr);
}

static void
usage()
{
  fprintf(stderr, "Usage: zfp <options>\n");
  fprintf(stderr, "General options:\n");
  fprintf(stderr, "  -h : read/write array and compression parameters from/to compressed header\n");
  fprintf(stderr, "  -q : quiet mode; suppress output\n");
  fprintf(stderr, "  -s : print error statistics\n");
  fprintf(stderr, "Input and output:\n");
  fprintf(stderr, "  -i <path> : uncompressed binary input file (\"-\" for stdin)\n");
  fprintf(stderr, "  -o <path> : decompressed binary output file (\"-\" for stdout)\n");
  fprintf(stderr, "  -z <path> : compressed input (w/o -i) or output file (\"-\" for stdin/stdout)\n");
  fprintf(stderr, "Array type and dimensions (needed with -i):\n");
  fprintf(stderr, "  -f : single precision (float type)\n");
  fprintf(stderr, "  -d : double precision (double type)\n");
  fprintf(stderr, "  -1 <nx> : 1D array dimensions\n");
  fprintf(stderr, "  -2 <nx> <ny> : 2D array dimensions\n");
  fprintf(stderr, "  -3 <nx> <ny> <nz> : 3D array dimensions\n");
  fprintf(stderr, "Compression parameters (needed with -i):\n");
  fprintf(stderr, "  -r <rate> : fixed rate (# compressed bits per floating-point value)\n");
  fprintf(stderr, "  -p <precision> : fixed precision (# uncompressed bits per value)\n");
  fprintf(stderr, "  -a <tolerance> : fixed accuracy (absolute error tolerance)\n");
  fprintf(stderr, "  -c <minbits> <maxbits> <maxprec> <minexp> : advanced usage\n");
  fprintf(stderr, "      minbits : min # bits per 4^d values in d dimensions\n");
  fprintf(stderr, "      maxbits : max # bits per 4^d values in d dimensions (0 for unlimited)\n");
  fprintf(stderr, "      maxprec : max # bits of precision per value (0 for full)\n");
  fprintf(stderr, "      minexp : min bit plane # coded (-1074 for all bit planes)\n");
  fprintf(stderr, "Examples:\n");
  fprintf(stderr, "  -i file : read uncompressed file and compress to memory\n");
  fprintf(stderr, "  -z file : read compressed file and decompress to memory\n");
  fprintf(stderr, "  -i ifile -z zfile : read uncompressed ifile, write compressed zfile\n");
  fprintf(stderr, "  -z zfile -o ofile : read compressed zfile, write decompressed ofile\n");
  fprintf(stderr, "  -i ifile -o ofile : read ifile, compress, decompress, write ofile\n");
  fprintf(stderr, "  -i file -s : read uncompressed file, compress to memory, print stats\n");
  fprintf(stderr, "  -i - -o - -s : read stdin, compress, decompress, write stdout, print stats\n");
  fprintf(stderr, "  -f -3 100 100 100 -r 16 : 2x fixed-rate compression of 100x100x100 floats\n");
  fprintf(stderr, "  -d -1 1000000 -r 32 : 2x fixed-rate compression of 1M doubles\n");
  fprintf(stderr, "  -d -2 1000 1000 -p 32 : 32-bit precision compression of 1000x1000 doubles\n");
  fprintf(stderr, "  -d -1 1000000 -a 1e-9 : compression of 1M doubles with < 1e-9 max error\n");
  fprintf(stderr, "  -d -1 1000000 -c 64 64 0 -1074 : 4x fixed-rate compression of 1M doubles\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char* argv[])
{
  /* default settings */
  zfp_type type = zfp_type_none;
  size_t typesize = 0;
  uint dims = 0;
  uint nx = 0;
  uint ny = 0;
  uint nz = 0;
  double rate = 0;
  uint precision = 0;
  double tolerance = 0;
  uint minbits = ZFP_MIN_BITS;
  uint maxbits = ZFP_MAX_BITS;
  uint maxprec = ZFP_MAX_PREC;
  int minexp = ZFP_MIN_EXP;
  int header = 0;
  int quiet = 0;
  int stats = 0;
  char* inpath = 0;
  char* zfppath = 0;
  char* outpath = 0;
  char mode = 0;

  /* local variables */
  int i;
  zfp_field* field = NULL;
  zfp_stream* zfp = NULL;
  bitstream* stream = NULL;
  void* fi = NULL;
  void* fo = NULL;
  void* buffer = NULL;
  size_t rawsize = 0;
  size_t zfpsize = 0;
  size_t bufsize = 0;

  /* parse command-line arguments */
  for (i = 1; i < argc; i++) {
    if (argv[i][0] != '-' || argv[i][2])
      usage();
    switch (argv[i][1]) {
      case '1':
        if (++i == argc || sscanf(argv[i], "%u", &nx) != 1)
          usage();
        ny = nz = 1;
        dims = 1;
        break;
      case '2':
        if (++i == argc || sscanf(argv[i], "%u", &nx) != 1 ||
            ++i == argc || sscanf(argv[i], "%u", &ny) != 1)
          usage();
        nz = 1;
        dims = 2;
        break;
      case '3':
        if (++i == argc || sscanf(argv[i], "%u", &nx) != 1 ||
            ++i == argc || sscanf(argv[i], "%u", &ny) != 1 ||
            ++i == argc || sscanf(argv[i], "%u", &nz) != 1)
          usage();
        dims = 3;
        break;
      case 'a':
        if (++i == argc || sscanf(argv[i], "%lf", &tolerance) != 1)
          usage();
        mode = 'a';
        break;
      case 'c':
        if (++i == argc || sscanf(argv[i], "%u", &minbits) != 1 ||
            ++i == argc || sscanf(argv[i], "%u", &maxbits) != 1 ||
            ++i == argc || sscanf(argv[i], "%u", &maxprec) != 1 ||
            ++i == argc || sscanf(argv[i], "%d", &minexp) != 1)
          usage();
        mode = 'c';
        break;
      case 'd':
        type = zfp_type_double;
        typesize = sizeof(double);
        break;
      case 'f':
        type = zfp_type_float;
        typesize = sizeof(float);
        break;
      case 'h':
        header = 1;
        break;
      case 'i':
        if (++i == argc)
          usage();
        inpath = argv[i];
        break;
      case 'o':
        if (++i == argc)
          usage();
        outpath = argv[i];
        break;
      case 'p':
        if (++i == argc || sscanf(argv[i], "%u", &precision) != 1)
          usage();
        mode = 'p';
        break;
      case 'q':
        quiet = 1;
        break;
      case 'r':
        if (++i == argc || sscanf(argv[i], "%lf", &rate) != 1)
          usage();
        mode = 'r';
        break;
      case 's':
        stats = 1;
        break;
      case 'z':
        if (++i == argc)
          usage();
        zfppath = argv[i];
        break;
      default:
        usage();
        break;
    }
  }

  /* make sure we have an input file */
  if (!inpath && !zfppath) {
    fprintf(stderr, "must specify uncompressed or compressed input file via -i or -z\n");
    return EXIT_FAILURE;
  }

  /* make sure we know floating-point type */
  if ((inpath || !header) && !typesize) {
    fprintf(stderr, "must specify scalar type via -f or -d\n");
    return EXIT_FAILURE;
  }

  /* make sure we know array dimensions */
  if ((inpath || !header) && !dims) {
    fprintf(stderr, "must specify array dimensions via -1, -2, or -3\n");
    return EXIT_FAILURE;
  }

  /* make sure we know (de)compression mode and parameters */
  if ((inpath || !header) && !mode) {
    fprintf(stderr, "must specify compression parameters via -a, -c, -p, or -r\n");
    return EXIT_FAILURE;
  }

  /* make sure we have input file for stats */
  if (stats && !inpath) {
    fprintf(stderr, "must specify input file via -i to compute stats\n");
    return EXIT_FAILURE;
  }

  /* make sure meta data comes from header or command line, not both */
  if (!inpath && zfppath && header && (typesize || dims)) {
    fprintf(stderr, "cannot specify both field type/size and header\n");
    return EXIT_FAILURE;
  }

  zfp = zfp_stream_open(NULL);
  field = zfp_field_alloc();

  /* read uncompressed or compressed file */
  if (inpath) {
    /* read uncompressed input file */
    FILE* file = !strcmp(inpath, "-") ? stdin : fopen(inpath, "rb");
    if (!file) {
      fprintf(stderr, "cannot open input file\n");
      return EXIT_FAILURE;
    }
    rawsize = typesize * nx * ny * nz;
    fi = malloc(rawsize);
    if (!fi) {
      fprintf(stderr, "cannot allocate memory\n");
      return EXIT_FAILURE;
    }
    if (fread(fi, typesize, nx * ny * nz, file) != nx * ny * nz) {
      fprintf(stderr, "cannot read input file\n");
      return EXIT_FAILURE;
    }
    fclose(file);
    zfp_field_set_pointer(field, fi);
  }
  else {
    /* read compressed input file in increasingly large chunks */
    FILE* file = !strcmp(zfppath, "-") ? stdin : fopen(zfppath, "rb");
    if (!file) {
      fprintf(stderr, "cannot open compressed file\n");
      return EXIT_FAILURE;
    }
    bufsize = 0x100;
    do {
      bufsize *= 2;
      buffer = realloc(buffer, bufsize);
      if (!buffer) {
        fprintf(stderr, "cannot allocate memory\n");
        return EXIT_FAILURE;
      }
      zfpsize += fread((uchar*)buffer + zfpsize, 1, bufsize - zfpsize, file);
    } while (zfpsize == bufsize);
    if (ferror(file)) {
      fprintf(stderr, "cannot read compressed file\n");
      return EXIT_FAILURE;
    }
    fclose(file);

    /* associate bit stream with buffer */
    stream = stream_open(buffer, bufsize);
    if (!stream) {
      fprintf(stderr, "cannot open compressed stream\n");
      return EXIT_FAILURE;
    }
    zfp_stream_set_bit_stream(zfp, stream);
  }

  /* set field dimensions and (de)compression parameters */
  if (inpath || !header) {
    /* initialize uncompressed field */
    zfp_field_set_type(field, type);
    switch (dims) {
      case 1:
        zfp_field_set_size_1d(field, nx);
        break;
      case 2:
        zfp_field_set_size_2d(field, nx, ny);
        break;
      case 3:
        zfp_field_set_size_3d(field, nx, ny, nz);
        break;
    }

    /* set (de)compression mode */
    switch (mode) {
      case 'a':
        zfp_stream_set_accuracy(zfp, tolerance, type);
        break;
      case 'p':
        zfp_stream_set_precision(zfp, precision, type);
        break;
      case 'r':
        zfp_stream_set_rate(zfp, rate, type, dims, 0);
        break;
      case 'c':
        if (!maxbits)
          maxbits = ZFP_MAX_BITS;
        if (!maxprec)
          maxprec = zfp_field_precision(field);
        if (!zfp_stream_set_params(zfp, minbits, maxbits, maxprec, minexp)) {
          fprintf(stderr, "invalid compression parameters\n");
          return EXIT_FAILURE;
        }
        break;
    }
  }

  /* compress input file if provided */
  if (inpath) {
    /* allocate buffer for compressed data */
    bufsize = zfp_stream_maximum_size(zfp, field);
    if (!bufsize) {
      fprintf(stderr, "invalid compression parameters\n");
      return EXIT_FAILURE;
    }
    buffer = malloc(bufsize);
    if (!buffer) {
      fprintf(stderr, "cannot allocate memory\n");
      return EXIT_FAILURE;
    }

    /* associate compressed bit stream with memory buffer */
    stream = stream_open(buffer, bufsize);
    if (!stream) {
      fprintf(stderr, "cannot open compressed stream\n");
      return EXIT_FAILURE;
    }
    zfp_stream_set_bit_stream(zfp, stream);

    /* optionally write header */
    if (header && !zfp_write_header(zfp, field, ZFP_HEADER_FULL)) {
      fprintf(stderr, "cannot write header\n");
      return EXIT_FAILURE;
    }

    /* compress data */
    zfpsize = zfp_compress(zfp, field);
    if (zfpsize == 0) {
      fprintf(stderr, "compression failed\n");
      return EXIT_FAILURE;
    }

    /* optionally write compressed data */
    if (zfppath) {
      FILE* file = !strcmp(zfppath, "-") ? stdout : fopen(zfppath, "wb");
      if (!file) {
        fprintf(stderr, "cannot create compressed file\n");
        return EXIT_FAILURE;
      }
      if (fwrite(buffer, 1, zfpsize, file) != zfpsize) {
        fprintf(stderr, "cannot write compressed file\n");
        return EXIT_FAILURE;
      }
      fclose(file);
    }
  }

  /* decompress data if necessary */
  if ((!inpath && zfppath) || outpath || stats) {
    /* obtain metadata from header when present */
    zfp_stream_rewind(zfp);
    if (header) {
      if (!zfp_read_header(zfp, field, ZFP_HEADER_FULL)) {
        fprintf(stderr, "incorrect or missing header\n");
        return EXIT_FAILURE;
      }
      type = field->type;
      switch (type) {
        case zfp_type_float:
          typesize = sizeof(float);
          break;
        case zfp_type_double:
          typesize = sizeof(double);
          break;
        default:
          fprintf(stderr, "unsupported type\n");
          return EXIT_FAILURE;
      }
      nx = MAX(field->nx, 1u);
      ny = MAX(field->ny, 1u);
      nz = MAX(field->nz, 1u);
    }

    /* allocate memory for decompressed data */
    rawsize = typesize * nx * ny * nz;
    fo = malloc(rawsize);
    if (!fo) {
      fprintf(stderr, "cannot allocate memory\n");
      return EXIT_FAILURE;
    }
    zfp_field_set_pointer(field, fo);

    /* decompress data */
    if (!zfp_decompress(zfp, field)) {
      fprintf(stderr, "decompression failed\n");
      return EXIT_FAILURE;
    }

    /* optionally write reconstructed data */
    if (outpath) {
      FILE* file = !strcmp(outpath, "-") ? stdout : fopen(outpath, "wb");
      if (!file) {
        fprintf(stderr, "cannot create output file\n");
        return EXIT_FAILURE;
      }
      if (fwrite(fo, typesize, nx * ny * nz, file) != nx * ny * nz) {
        fprintf(stderr, "cannot write output file\n");
        return EXIT_FAILURE;
      }
      fclose(file);
    }
  }

  /* print compression and error statistics */
  if (!quiet) {
    fprintf(stderr, "type=%s nx=%u ny=%u nz=%u", type == zfp_type_float ? "float" : "double", nx, ny, nz);
    fprintf(stderr, " raw=%lu zfp=%lu ratio=%.3g rate=%.4g", (unsigned long)rawsize, (unsigned long)zfpsize, (double)rawsize / zfpsize, CHAR_BIT * (double)zfpsize / (nx * ny * nz));
    if (stats)
      print_error(fi, fo, type, nx * ny * nz);
    fprintf(stderr, "\n");
  }

  /* free allocated storage */
  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(stream);
  free(buffer);
  free(fi);
  free(fo);

  return EXIT_SUCCESS;
}
