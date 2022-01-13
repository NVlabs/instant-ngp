#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include "zfp.h"
#include "zfparray1.h"
#include "zfparray2.h"
#include "zfparray3.h"

// size of arrays to test
#ifdef TESTZFP_LARGE_ARRAYS
  #define TEST_SIZE 16 // 16^6 = 16 M scalars
  #define WITH_TIMINGS
#elif defined(TESTZFP_MEDIUM_ARRAYS)
  #define TEST_SIZE 8 // 8^6 = 256 K scalars
#else
  #define TEST_SIZE 4 // 4^6 = 4096 scalars
  #include "fields.h"
#endif

typedef unsigned char uchar;
typedef unsigned long long uint64;

int width = 72; // characters per line

// polynomial x - 3 x^2 + 4 x^4
template <typename Scalar>
inline Scalar
polynomial(volatile Scalar x)
{
  // volatile used to ensure bit-for-bit reproducibility across compilers
  volatile Scalar xx = x * x;
  volatile Scalar yy = 4 * xx - 3;
  volatile Scalar p = x + xx * yy;
  return p;
}

// initialize array
template <typename Scalar>
inline void
initialize(Scalar* p, int nx, int ny, int nz, Scalar (*f)(Scalar))
{
  nx = std::max(nx, 1);
  ny = std::max(ny, 1);
  nz = std::max(nz, 1);
#if TEST_SIZE == 4
  // use precomputed small arrays for portability
  uint d = nz == 1 ? ny == 1 ? 0 : 1 : 2;
  std::copy(&Field<Scalar>::array[d][0], &Field<Scalar>::array[d][0] + nx * ny * nz, p);
#else
  for (int k = 0; k < nz; k++) {
    volatile Scalar z = Scalar(2 * k - nz + 1) / nz;
    volatile Scalar fz = nz > 1 ? f(z) : Scalar(1);
    for (int j = 0; j < ny; j++) {
      volatile Scalar y = Scalar(2 * j - ny + 1) / ny;
      volatile Scalar fy = ny > 1 ? f(y) : Scalar(1);
      for (int i = 0; i < nx; i++) {
        volatile Scalar x = Scalar(2 * i - nx + 1) / nx;
        volatile Scalar fx = nx > 1 ? f(x) : Scalar(1);
        *p++ = fx * fy * fz;
      }
    }
  }
#endif
}

// compute checksum
inline uint32
hash(const void* p, size_t n)
{
  uint32 h = 0;
  for (const uchar* q = static_cast<const uchar*>(p); n; q++, n--) {
    // Jenkins one-at-a-time hash; see http://www.burtleburtle.net/bob/hash/doobs.html
    h += *q;
    h += h << 10;
    h ^= h >>  6;
  }
  h += h <<  3;
  h ^= h >> 11;
  h += h << 15;
  return h;
}

// test fixed-rate mode
template <typename Scalar>
inline uint
test_rate(zfp_stream* stream, const zfp_field* input, double rate, Scalar tolerance)
{
  uint failures = 0;
  uint n = zfp_field_size(input, NULL);
  uint dims = zfp_field_dimensionality(input);
  zfp_type type = zfp_field_type(input);

  // allocate memory for compressed data
  rate = zfp_stream_set_rate(stream, rate, type, dims, 0);
  size_t bufsize = zfp_stream_maximum_size(stream, input);
  uchar* buffer = new uchar[bufsize];
  bitstream* s = stream_open(buffer, bufsize);
  zfp_stream_set_bit_stream(stream, s);

  // perform compression test
  std::ostringstream status;
  status << "  compress:  ";
  status << " rate=" << std::fixed << std::setprecision(0) << std::setw(2) << rate;
#ifdef WITH_TIMINGS
  clock_t c = clock();
#endif
  zfp_stream_rewind(stream);
  size_t outsize = zfp_compress(stream, input);
#ifdef WITH_TIMINGS
  double time = double(clock() - c) / CLOCKS_PER_SEC;
  double throughput = (n * sizeof(Scalar)) / (0x100000 * time);
  status << " throughput=" << std::setprecision(1) << std::setw(6) << throughput << " MB/s";
#endif
  bool pass = true;
  // make sure compressed size matches rate
  size_t bytes = (size_t)floor(rate * zfp_field_size(input, NULL) / CHAR_BIT + 0.5);
  if (outsize != bytes) {
    status << " [" << outsize << " != " << bytes << "]";
    pass = false;
  }
  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  // perform decompression test
  status.str("");
  status << "  decompress:";
  status << " rate=" << std::fixed << std::setprecision(0) << std::setw(2) << rate;
  Scalar* g = new Scalar[n];
  zfp_field* output = zfp_field_alloc();
  *output = *input;
  zfp_field_set_pointer(output, g);
#ifdef WITH_TIMINGS
  c = clock();
#endif
  zfp_stream_rewind(stream);
  pass = zfp_decompress(stream, output);
  if (!pass)
    status << " [decompression failed]";
  else {
#ifdef WITH_TIMINGS
    double time = double(clock() - c) / CLOCKS_PER_SEC;
    double throughput = (n * sizeof(Scalar)) / (0x100000 * time);
    status << " throughput=" << std::setprecision(1) << std::setw(6) << throughput << " MB/s";
#endif
    // compute max error
    Scalar* f = static_cast<Scalar*>(zfp_field_pointer(input));
    Scalar emax = 0;
    for (uint i = 0; i < n; i++)
      emax = std::max(emax, std::abs(f[i] - g[i]));
    status << std::scientific;
    status.precision(3);
    // make sure max error is within tolerance
    if (emax <= tolerance)
      status << " " << emax << " <= " << tolerance;
    else {
      status << " [" << emax << " > " << tolerance << "]";
      pass = false;
    }
  }
  zfp_field_free(output);
  delete[] g;
  stream_close(s);
  delete[] buffer;
  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  return failures;
}

// test fixed-precision mode
template <typename Scalar>
inline uint
test_precision(zfp_stream* stream, const zfp_field* input, uint precision, size_t bytes)
{
  uint failures = 0;
  uint n = zfp_field_size(input, NULL);
  zfp_type type = zfp_field_type(input);

  // allocate memory for compressed data
  zfp_stream_set_precision(stream, precision, type);
  size_t bufsize = zfp_stream_maximum_size(stream, input);
  uchar* buffer = new uchar[bufsize];
  bitstream* s = stream_open(buffer, bufsize);
  zfp_stream_set_bit_stream(stream, s);

  // perform compression test
  std::ostringstream status;
  status << "  compress:  ";
  status << " precision=" << std::setw(2) << precision;
  zfp_stream_rewind(stream);
  size_t outsize = zfp_compress(stream, input);
  double ratio = double(n * sizeof(Scalar)) / outsize;
  status << " ratio=" << std::fixed << std::setprecision(3) << std::setw(7) << ratio;
  bool pass = true;
  // make sure compressed size agrees
  if (outsize != bytes) {
    status << " [" << outsize << " != " << bytes << "]";
    pass = false;
  }
  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  // perform decompression test
  status.str("");
  status << "  decompress:";
  status << " precision=" << std::setw(2) << precision;
  Scalar* g = new Scalar[n];
  zfp_field* output = zfp_field_alloc();
  *output = *input;
  zfp_field_set_pointer(output, g);
  zfp_stream_rewind(stream);
  pass = zfp_decompress(stream, output);
  if (!pass)
    status << " [decompression failed]";
  zfp_field_free(output);
  delete[] g;
  stream_close(s);
  delete[] buffer;
  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  return failures;
}

// test fixed-accuracy mode
template <typename Scalar>
inline uint
test_accuracy(zfp_stream* stream, const zfp_field* input, Scalar tolerance, size_t bytes)
{
  uint failures = 0;
  uint n = zfp_field_size(input, NULL);
  zfp_type type = zfp_field_type(input);

  // allocate memory for compressed data
  tolerance = zfp_stream_set_accuracy(stream, tolerance, type);
  size_t bufsize = zfp_stream_maximum_size(stream, input);
  uchar* buffer = new uchar[bufsize];
  bitstream* s = stream_open(buffer, bufsize);
  zfp_stream_set_bit_stream(stream, s);

  // perform compression test
  std::ostringstream status;
  status << "  compress:  ";
  status << " tolerance=" << std::scientific << std::setprecision(3) << tolerance;
  zfp_stream_rewind(stream);
  size_t outsize = zfp_compress(stream, input);
  double ratio = double(n * sizeof(Scalar)) / outsize;
  status << " ratio=" << std::fixed << std::setprecision(3) << std::setw(7) << ratio;
  bool pass = true;
  // make sure compressed size agrees
  if (outsize != bytes) {
    status << " [" << outsize << " != " << bytes << "]";
    pass = false;
  }
  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  // perform decompression test
  status.str("");
  status << "  decompress:";
  status << " tolerance=" << std::scientific << std::setprecision(3) << tolerance;
  Scalar* g = new Scalar[n];
  zfp_field* output = zfp_field_alloc();
  *output = *input;
  zfp_field_set_pointer(output, g);
  zfp_stream_rewind(stream);
  pass = zfp_decompress(stream, output);
  if (!pass)
    status << " [decompression failed]";
  else {
    // compute max error
    Scalar* f = static_cast<Scalar*>(zfp_field_pointer(input));
    Scalar emax = 0;
    for (uint i = 0; i < n; i++)
      emax = std::max(emax, std::abs(f[i] - g[i]));
    status << std::scientific << std::setprecision(3) << " ";
    // make sure max error is within tolerance
    if (emax <= tolerance)
      status << emax << " <= " << tolerance;
    else if (tolerance == 0)
      status << "(" << emax << " > 0)";
    else {
      status << "[" << emax << " > " << tolerance << "]";
      pass = false;
    }
  }
  zfp_field_free(output);
  delete[] g;
  stream_close(s);
  delete[] buffer;
  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  return failures;
}

// perform 1D differencing
template <typename Scalar>
inline void
update_array1(zfp::array1<Scalar>& a)
{
  for (uint i = 0; i < a.size() - 1; i++)
    a(i) -= a(i + 1);
  for (uint i = 0; i < a.size() - 1; i++)
    a(0) = std::max(a(0), a(i));
}

// perform 2D differencing
template <typename Scalar>
inline void
update_array2(zfp::array2<Scalar>& a)
{
  for (uint j = 0; j < a.size_y(); j++)
    for (uint i = 0; i < a.size_x() - 1; i++)
      a(i, j) -= a(i + 1, j);
  for (uint j = 0; j < a.size_y() - 1; j++)
    for (uint i = 0; i < a.size_x(); i++)
      a(i, j) -= a(i, j + 1);
  for (uint j = 0; j < a.size_y() - 1; j++)
    for (uint i = 0; i < a.size_x() - 1; i++)
      a(0, 0) = std::max(a(0, 0), a(i, j));
}

// perform 3D differencing
template <typename Scalar>
inline void
update_array3(zfp::array3<Scalar>& a)
{
  for (uint k = 0; k < a.size_z(); k++)
    for (uint j = 0; j < a.size_y(); j++)
      for (uint i = 0; i < a.size_x() - 1; i++)
        a(i, j, k) -= a(i + 1, j, k);
  for (uint k = 0; k < a.size_z(); k++)
    for (uint j = 0; j < a.size_y() - 1; j++)
      for (uint i = 0; i < a.size_x(); i++)
        a(i, j, k) -= a(i, j + 1, k);
  for (uint k = 0; k < a.size_z() - 1; k++)
    for (uint j = 0; j < a.size_y(); j++)
      for (uint i = 0; i < a.size_x(); i++)
        a(i, j, k) -= a(i, j, k + 1);
  for (uint k = 0; k < a.size_z() - 1; k++)
    for (uint j = 0; j < a.size_y() - 1; j++)
      for (uint i = 0; i < a.size_x() - 1; i++)
        a(0, 0, 0) = std::max(a(0, 0, 0), a(i, j, k));
}

template <class Array>
inline void update_array(Array& a);

template <>
inline void
update_array(zfp::array1<float>& a) { update_array1(a); }

template <>
inline void
update_array(zfp::array1<double>& a) { update_array1(a); }

template <>
inline void
update_array(zfp::array2<float>& a) { update_array2(a); }

template <>
inline void
update_array(zfp::array2<double>& a) { update_array2(a); }

template <>
inline void
update_array(zfp::array3<float>& a) { update_array3(a); }

template <>
inline void
update_array(zfp::array3<double>& a) { update_array3(a); }

// test random-accessible array primitive
template <class Array, typename Scalar>
inline uint
test_array(Array& a, const Scalar* f, uint n, double tolerance, double dfmax)
{
  uint failures = 0;

  // test construction
  std::ostringstream status;
  status << "  construct: ";
  Scalar emax = 0;
  for (uint i = 0; i < n; i++)
    emax = std::max(emax, std::abs(f[i] - a[i]));
  status << std::scientific;
  status.precision(3);
  // make sure max error is within tolerance
  bool pass = true;
  if (emax <= tolerance)
    status << " " << emax << " <= " << tolerance;
  else {
    status << " [" << emax << " > " << tolerance << "]";
    pass = false;
  }

  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  // test array updates
  status.str("");
  status << "  update:    ";
  update_array(a);
  Scalar amax = a[0];
  pass = true;
  if (std::abs(amax - dfmax) <= 1e-3 * dfmax)
    status << " " << amax << " ~ " << dfmax;
  else {
    status << " [" << amax << " != " << dfmax << "]";
    pass = false;
  }

  std::cout << std::setw(width) << std::left << status.str() << (pass ? " OK " : "FAIL") << std::endl;
  if (!pass)
    failures++;

  return failures;
}

// test arrays with m^6 scalars
template <typename Scalar>
inline uint
test(uint m)
{
  uint failures = 0;
  uint n = m * m * m * m * m * m;
  Scalar* f = new Scalar[n];
  // test 1D, 2D, and 3D arrays
  for (uint d = 1; d <= 3; d++) {
    // determine array size
    uint nx, ny, nz;
    zfp_field* field = zfp_field_alloc();
    zfp_field_set_type(field, zfp::codec<Scalar>::type);
    zfp_field_set_pointer(field, f);
    switch (d) {
      case 1:
        nx = n;
        ny = 0;
        nz = 0;
        zfp_field_set_size_1d(field, nx);
        break;
      case 2:
        nx = m * m * m;
        ny = m * m * m;
        nz = 0;
        zfp_field_set_size_2d(field, nx, ny);
        break;
      case 3:
        nx = m * m;
        ny = m * m;
        nz = m * m;
        zfp_field_set_size_3d(field, nx, ny, nz);
        break;
    }
    initialize<Scalar>(f, nx, ny, nz, polynomial);
    uint t = (zfp_field_type(field) == zfp_type_float ? 0 : 1);
    std::cout << "testing " << d << "D array of " << (t == 0 ? "floats" : "doubles") << std::endl;

    // test data integrity
    uint32 checksum[2][3] = {
#if TEST_SIZE == 4
      { 0xdad6fd69u, 0x000f8df1u, 0x60993f48u },
      { 0x8d95b1fdu, 0x96a0e601u, 0x66e77c83u },
#elif TEST_SIZE == 8
      { 0x269fb420u, 0xfc4fd405u, 0x733b9643u },
      { 0x3321e28bu, 0xfcb8f0f0u, 0xd0f6d6adu },
#elif TEST_SIZE == 16
      { 0x62d6c2b5u, 0x88aa838eu, 0x84f98253u },
      { 0xf2bd03a4u, 0x10084595u, 0xb8df0e02u },
#endif
    };
    uint32 h = hash(f, n * sizeof(Scalar));
    if (h != checksum[t][d - 1])
      std::cout << "warning: array checksum mismatch; tests below may fail" << std::endl;

    // open compressed stream
    zfp_stream* stream = zfp_stream_open(0);

    // test fixed rate
    for (uint rate = 2u >> t, i = 0; rate <= 32 * (t + 1); rate *= 4, i++) {
      // expected max errors
#if TEST_SIZE == 4
      Scalar emax[2][3][4] = {
        {
          {1.998e+00, 7.767e-03, 0.000e+00},
          {2.356e-01, 3.939e-04, 7.451e-09},
          {2.479e-01, 1.525e-03, 7.451e-08},
        },
        {
          {1.998e+00, 9.976e-01, 1.360e-05},
          {2.944e+00, 2.491e-02, 2.578e-06},
          {6.103e-01, 3.253e-02, 6.467e-06},
        }
      };
#elif TEST_SIZE == 8
      Scalar emax[2][3][4] = {
        {
          {2.000e+00, 1.425e-03, 0.000e+00},
          {7.110e-02, 1.264e-05, 2.329e-10},
          {1.864e-02, 2.814e-05, 1.193e-07},
        },
        {
          {2.000e+00, 1.001e+00, 3.084e-06, 0.000e+00},
          {2.266e+00, 3.509e-03, 1.784e-08, 0.000e+00},
          {2.494e-01, 1.473e-03, 7.060e-08, 3.470e-18},
        }
      };
#elif TEST_SIZE == 16
      Scalar emax[2][3][4] = {
        {
          {2.000e+00, 1.304e-03, 0.000e+00},
          {6.907e-02, 5.961e-07, 2.911e-11},
          {3.458e-03, 7.153e-07, 2.385e-07},
        },
        {
          {2.000e+00, 1.001e+00, 6.353e-07, 0.000e+00},
          {2.036e+00, 3.174e-04, 1.646e-10, 5.294e-23},
          {5.483e-02, 8.559e-05, 4.564e-10, 8.674e-19},
        }
      };
#endif
      failures += test_rate<Scalar>(stream, field, rate, emax[t][d - 1][i]);
    }

    if (stream_word_bits != 64)
      std::cout << "warning: stream word size is smaller than 64; tests below may fail" << std::endl;

    // test fixed precision
    for (uint prec = 4u << t, i = 0; i < 3; prec *= 2, i++) {
      // expected compressed sizes
#if TEST_SIZE == 4
      size_t bytes[2][3][3] = {
        {
          {2176, 3256, 6272},
          { 576, 1296, 4136},
          { 128,  720, 4096},
        },
        {
          {3640, 6656, 14576},
          {1392, 4232, 12312},
          { 744, 4120, 12304},
        },
      };
#elif TEST_SIZE == 8
      size_t bytes[2][3][3] = {
        {
          {138864, 204456, 349888},
          { 35216,  63632, 163008},
          {  8856,  26768, 133360},
        },
        {
          {229048, 374264, 786504},
          { 69776, 169168, 564192},
          { 28304, 134904, 588600},
        },
      };
#elif TEST_SIZE == 16
      size_t bytes[2][3][3] = {
        {
          {8886920, 13080944, 21487696},
          {2240256,  3457592,  7787752},
          { 570656,  1277128,  4803216},
        },
        {
          {14654848, 23059592, 45965208},
          { 3850784,  8168520, 25149520},
          { 1375440,  4901552, 24339800},
        },
      };
#endif
      failures += test_precision<Scalar>(stream, field, prec, bytes[t][d - 1][i]);
    }

    // test fixed accuracy
    for (uint i = 0; i < 3; i++) {
      Scalar tol[] = { 1e-3, 2 * std::numeric_limits<Scalar>::epsilon(), 0 };
      // expected compressed sizes
#if TEST_SIZE == 4
      size_t bytes[2][3][3] = {
        {
          {4752, 10184, 14192},
          {2920,  8720, 12216},
          {4264, 10408, 12280},
        },
        {
          {5136, 25416, 30960},
          {3016, 23664, 28696},
          {4288, 25280, 28688},
        },
      };
#elif TEST_SIZE == 8
      size_t bytes[2][3][3] = {
        {
          {272208, 552856, 792528},
          {105440, 329280, 572552},
          { 90584, 381264, 588528},
        },
        {
          {296672, 1478648, 1834416},
          {111560, 1250056, 1609720},
          { 92120, 1327544, 1637168},
        },
      };
#elif TEST_SIZE == 16
      size_t bytes[2][3][3] = {
        {
          {17416688, 32229448, 47431656},
          { 5327000, 14827440, 28920960},
          { 2721504, 12688136, 26308832},
        },
        {
          {18982344, 85195360, 107965448},
          { 5715280, 63417800,  86969224},
          { 2819224, 66345592,  89157296},
        },
      };
#endif
      failures += test_accuracy<Scalar>(stream, field, tol[i], bytes[t][d - 1][i]);
    }

    // test compressed array support
#if TEST_SIZE == 4
    Scalar emax[2][3] = {
      {0.000e+00, 7.451e-09, 7.451e-08},
      {2.354e-10, 5.731e-09, 2.804e-08},
    };
    Scalar dfmax[2][3] = {
      {4.385e-03, 9.260e-02, 3.760e-01},
      {4.385e-03, 9.260e-02, 3.760e-01},
    };
#elif TEST_SIZE == 8
    Scalar emax[2][3] = {
      {0.000e+00, 2.329e-10, 1.193e-07},
      {1.302e-11, 5.678e-11, 4.148e-10},
    };
    Scalar dfmax[2][3] = {
      {6.866e-05, 1.792e-03, 2.239e-02},
      {6.866e-05, 1.792e-03, 2.239e-02},
    };
#elif TEST_SIZE == 16
    Scalar emax[2][3] = {
      {0.000e+00, 2.911e-11, 2.385e-07},
      {4.464e-13, 3.051e-13, 1.224e-12},
    };
    Scalar dfmax[2][3] = {
      {1.073e-06, 2.906e-05, 4.714e-04},
      {1.073e-06, 2.880e-05, 4.714e-04},
    };
#endif
    double rate = 24;
    switch (d) {
      case 1: {
          zfp::array1<Scalar> a(nx, rate, f);
          failures += test_array(a, f, n, emax[t][d - 1], dfmax[t][d - 1]);
        }
        break;
      case 2: {
          zfp::array2<Scalar> a(nx, ny, rate, f);
          failures += test_array(a, f, n, emax[t][d - 1], dfmax[t][d - 1]);
        }
        break;
      case 3: {
          zfp::array3<Scalar> a(nx, ny, nz, rate, f);
          failures += test_array(a, f, n, emax[t][d - 1], dfmax[t][d - 1]);
        }
        break;
    }

    std::cout << std::endl;
    zfp_stream_close(stream);
    zfp_field_free(field);
  }

  delete[] f;
  return failures;
}

int main()
{
  uint failures = 0;
  uint m = TEST_SIZE;
  failures += test<float>(m);
  failures += test<double>(m);
  if (failures)
    std::cout << failures << " test(s) failed" << std::endl;
  else
    std::cout << "all tests passed" << std::endl;
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
