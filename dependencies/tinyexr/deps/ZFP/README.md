ZFP
===

INTRODUCTION
------------

This is zfp 0.5.0, an open source C/C++ library for compressed numerical
arrays that support high throughput read and write random access.  zfp was
written by Peter Lindstrom at Lawrence Livermore National Laboratory, and
is loosely based on the algorithm described in the following paper:

    Peter Lindstrom
    "Fixed-Rate Compressed Floating-Point Arrays"
    IEEE Transactions on Visualization and Computer Graphics,
      20(12):2674-2683, December 2014
    doi:10.1109/TVCG.2014.2346458

zfp was originally designed for floating-point data only, but has been
extended to also support integer data, and could for instance be used to
compress images and quantized volumetric data.  To achieve high compression
ratios, zfp uses lossy but optionally error-bounded compression.  Although
bit-for-bit lossless compression of floating-point data is not always
possible, zfp is usually accurate to within machine epsilon in near-lossless
mode.

zfp works best for 2D and 3D arrays that exhibit spatial coherence, such
as smooth fields from physics simulations, images, regularly sampled terrain
surfaces, etc.  Although zfp also provides a 1D array class that can be used
for 1D signals such as audio, or even unstructured floating-point streams,
the compression scheme has not been well optimized for this use case, and
rate and quality may not be competitive with floating-point compressors
designed specifically for 1D streams.

zfp is freely available as open source under a BSD license, as outlined in
the file 'LICENSE'.  For information on the API and general usage, please
see the file 'API' in this directory.


INSTALLATION
------------

zfp consists of three distinct parts: a compression library written in C;
a set of C++ header files that implement compressed arrays; and a set of
C and C++ examples.  The main compression codec is written in C and should
conform to both the ISO C89 and C99 standards.  The C++ array classes are
implemented entirely in header files and can be included as is, but since
they call the compression library applications must link with libzfp.

To compile libzfp and all example programs (see below for more details)
on Linux or OS X, type

    make

from this directory.  This compiles libzfp as a static library and the
example programs.  To optionally create a shared library, type

    make shared

and set LD_LIBRARY_PATH to point to ./lib.  To test the compressor, type

    make test

If the compilation or regression tests fail, it is possible that some of
the macros in the file 'Config' have to be adjusted.  Also, the tests may
fail due to minute differences in the computed floating-point fields
being compressed (as indicated by checksum errors).  It is surprisingly
difficult to portably generate a floating-point array that agrees
bit-for-bit across platforms.  If most tests succeed and the failures
result in byte sizes and error values reasonably close to the expected
values, then it is likely that the compressor is working correctly.

NOTE: zfp requires 64-bit compiler and operating system support.

zfp has successfully been built and tested using these compilers:

    gcc versions 4.4.7, 4.7.2, 4.8.2, 4.9.2, 5.1.0
    icc versions 12.0.5, 12.1.5, 15.0.4, 16.0.1
    clang versions 3.4.2, 3.6.0
    xlc version 12.1
    mingw32-gcc version 4.8.1


ALGORITHM OVERVIEW
------------------

The zfp lossy compression scheme is based on the idea of breaking a
d-dimensional array into independent blocks of 4^d values, e.g. 4x4x4
values in three dimensions.  Each block is compressed/decompressed
entirely independently from all other blocks.  In this sense, zfp is
similar to current hardware texture compression schemes for image
coding implemented on graphics cards and mobile devices.

The compression scheme implemented in this version of zfp has evolved
from the method described in the paper cited above, and can conceptually
be thought of as consisting of eight sequential steps (in practice some
steps are consolidated or exist only for illustrative purposes):

1. The d-dimensional array is partitioned into blocks of dimensions 4^d.
If the array dimensions are not multiples of four, then blocks near the
boundary are padded to the next multiple of four.  This padding is
invisible to the application.

2. The independent floating-point values in a block are converted to what
is known as a block-floating-point representation, which uses a single,
common floating-point exponent for all 4^d values.  The effect of this
conversion is to turn each floating-point value into a 31- or 63-bit
signed integer.  Note that this step is not performed if the input data
already consists of integers.

3. The integers are decorrelated using a custom, high-speed, near
orthogonal transform similar to the discrete cosine transform used in
JPEG image coding.  The transform exploits separability and is implemented
efficiently in-place using the lifting scheme, requiring only 2.5*d
integer additions and 1.5*d bit shifts by one per integer in d dimensions.
If the data is "smooth," then this transform will turn most integers into
small signed values clustered around zero.

4. The two's complement signed integers are converted to their negabinary
(base negative two) representation using one addition and one bit-wise
exclusive or per integer.  Because negabinary has no dedicated single sign
bit, these integers are subsequently treated as unsigned.

5. The unsigned integer coefficients are reordered in a manner similar to
JPEG zig-zag ordering so that statistically they appear in a roughly
monotonically decreasing order.  Coefficients corresponding to low
frequencies tend to have larger magnitude, and are listed first.  In 3D,
coefficients corresponding to frequencies i, j, k in the three dimensions
are ordered by i + j + k first, and then by i^2 + j^2 + k^2.

6. The bits that represent the list of 4^d integers are now ordered by
coefficient.  These bits are transposed so that they are instead ordered
by bit plane, from most to least significant bit.  Viewing each bit plane
as an integer, with the lowest bit corresponding to the lowest frequency
coefficient, the anticipation is that the first several of these transposed
integers are small, because the coefficients are assumed to be ordered by
magnitude.

7. The transform coefficients are compressed losslessly using embedded
coding by exploiting the property that the coefficients tend to have many
leading zeros that need not be encoded explicitly.  Each bit plane is
encoded in two parts, from lowest to highest bit.  First the n lowest bits
are emitted verbatim, where n depends on previous bit planes and is
initially zero.  Then a variable-length representation, x, of the
remaining 4^d - n bits is encoded.  For such an integer x, a single bit is
emitted to indicate if x = 0, in which case we are done with the current
bit plane.  If not, then bits of x are emitted, starting from the lowest
bit, until a one bit is emitted.  This triggers another test whether this
is the highest set bit of x, and the result of this test is output as a
single bit.  If not, then the procedure repeats until all n of x's value
bits have been output, where 2^(n-1) <= x < 2^n.  This can be thought of
as a run-length encoding of the zeros of x, where the run lengths are
expressed in unary.  The current value of n is then passed on to the next
bit plane, which is encoded by first emitting its n lowest bits.  The
assumption is that these bits correspond to n coefficients whose most
significant bits have already been output, i.e. these n bits are
essentially random and not compressible.  Following this, the remaining
4^d - n bits of the bit plane are run-length encoded as described above,
which potentially results in n being increased.

8. The embedded coder emits one bit at a time, with each successive bit
potentially improving the quality of the reconstructed signal.  The early
bits are most important and have the greatest impact on signal quality,
with the last few bits providing very small changes.  The resulting
compressed bit stream can be truncated at any point and still allow for a
valid approximate reconstruction of the original signal.  The final step
truncates the bit stream in one of three ways: to a fixed number of bits
(the fixed-rate mode); after some fixed number of bit planes have been
encoded (the fixed-precision mode); or until a lowest bit plane number has
been encoded, as expressed in relation to the common floating-point
exponent within the block (the fixed-accuracy mode).

Various parameters are exposed for controlling the quality and compressed
size of a block, and can be specified by the user at a very fine
granularity.  These parameters are discussed below.


CODE EXAMPLES
-------------

The 'examples' directory includes six programs that make use of the
compressor.

The 'simple' program is a minimal example that shows how to call the
compressor and decompressor on a double-precision 3D array.  Without
the '-d' option, it will compress the array and write the compressed
stream to standard output.  With the '-d' option, it will instead
read the compressed stream from standard input and decompress the
array:

    simple > compressed.zfp
    simple -d < compressed.zfp

For a more elaborate use of the compressor, see the 'zfp' example.

The 'diffusion' example is a simple forward Euler solver for the heat
equation on a 2D regular grid, and is intended to show how to declare
and work with zfp's compressed arrays, as well as give an idea of how
changing the compression rate affects the error in the solution.  The
usage is:

    diffusion-zfp [rate] [nx] [ny] [nt]

where 'rate' specifies the exact number of compressed bits to store per
double-precision floating-point value (default = 64); 'nx' and 'ny'
specify the grid size (default = 100x100); and 'nt' specifies the number
of time steps to run (the default is to run until time t = 1).

Running diffusion with the following arguments

    diffusion-zfp 8
    diffusion-zfp 12
    diffusion-zfp 20
    diffusion-zfp 64

should result in this output

    rate=8 sum=0.996442 error=4.813938e-07
    rate=12 sum=0.998338 error=1.967777e-07
    rate=20 sum=0.998326 error=1.967952e-07
    rate=64 sum=0.998326 error=1.967957e-07

For speed and quality comparison, diffusion-raw solves the same problem
using uncompressed double-precision arrays.

The 'zfp' program is primarily intended for evaluating the rate-distortion
(compression ratio and quality) provided by the compressor, but since
version 0.5.0 also allows reading and writing compressed data sets.  zfp
takes as input a raw, binary array of floats or doubles, and optionally
outputs a compressed or reconstructed array obtained after lossy
compression followed by decompression.  Various statistics on compression
rate and error are also displayed.

zfp requires a set of command-line options, the most important being the
-i option that specifies that the input is uncompressed.  When present,
"-i <file>" tells zfp to read the uncompressed input file and compress it
to memory.  If desired, the compressed stream can be written to an ouptut
file using "-z <file>".  When -i is absent, on the other hand, -z names
the compressed input (not output) file, which is then decompressed.  In
either case, "-o <file>" can be used to output the reconstructed array
resulting from lossy compression and decompression.

So, to compress a file, use "-i file.in -z file.zfp".  To later decompress
the file, use "-z file.zfp -o file.out".  A single dash "-" can be used in
place of a file name to denote standard input or output.

When reading uncompressed input, the floating-point precision (single or
double) must be specified using either -f (float) or -d (double).  In
addition, the array dimensions must be specified using "-1 nx" (for 1D
arrays), "-2 nx ny" (for 2D arrays), or "-3 nx ny nz" (for 3D arrays).
For multidimensional arrays, x varies faster than y, which in turn varies
faster than z.  That is, a 3D input file should correspond to a flattened
C array declared as a[nz][ny][nx].

Note that "-2 nx ny" is not equivalent to "-3 nx ny 1", even though the
same number of values are compressed.  One invokes the 2D codec, while the
other uses the 3D codec, which in this example has to pad the input to an
nx * ny * 4 array since arrays are partitioned into blocks of dimensions
4^d.  Such padding usually negatively impacts compression.

Using -h, the array dimensions and type are stored in a header of the
compressed stream so that they do not have to be specified on the command
line during decompression.  The header also stores compression parameters,
which are described below.

zfp accepts several options for specifying how the data is to be compressed.
The most general of these, the -c option, takes four constraint parameters
that together can be used to achieve various effects.  These constraints
are:

    minbits: the minimum number of bits used to represent a block
    maxbits: the maximum number of bits used to represent a block
    maxprec: the maximum number of bit planes encoded
    minexp:  the smallest bit plane number encoded

Options -r, -p, and -a provide a simpler interface to setting all of
the above parameters (see below).  Bit plane e refers to those bits whose
place value is 2^e.  For instance, in single precision, bit planes -149
through 127 are supported (when also counting denormalized numbers); for
double precision, bit planes -1074 through 1023 are supported.

Care must be taken to allow all constraints to be met, as encoding
terminates as soon as a single constraint is violated (except minbits,
which is satisfied at the end of encoding by padding zeros).  The effects
of the above four parameters are best explained in terms of the three main
compression modes supported by zfp (see Algorithm Overview above for
additional details):

  Fixed rate (option -r):
    In fixed-rate mode, each compressed block of 4^d values in d dimensions
    is stored using a fixed number of bits specified by the user.  This can
    be achieved using option -c by setting minbits = maxbits, maxprec = 64,
    and minexp = -1074.  The fixed-rate mode is needed to support random
    access to blocks, where the amortized number of bits used per value is
    given by rate = maxbits / 4^d.  Note that each block stores a leading
    all-zeros bit and common exponent, and maxbits must be at least 9 for
    single precision and 12 for double precision.

  Fixed precision (option -p):
    In fixed-precision mode, the number of bits used to encode a block may
    vary, but the number of bit planes (i.e. the precision) encoded for the
    transform coefficients is fixed.  This mode is achieved by specifying
    the precision in maxprec and fully relaxing the size constraints, i.e.
    minbits = 0, maxbits = 4171, and minexp = -1074.  Fixed-precision
    mode is preferable when relative rather than absolute errors matter.

  Fixed accuracy (option -a):
    In fixed-accuracy mode, all transform coefficient bit planes up to a
    minimum bit plane number are encoded.  (The actual minimum bit plane
    is not necessarily minexp, but depends on the dimensionality of the
    data.  The reason for this is that the inverse transform incurs range
    expansion, and the amount of expansion depends on the number of
    dimensions.)  Thus, minexp should be interpreted as the base-2 logarithm
    of an absolute error tolerance.  In other words, given an uncompressed
    value f and a reconstructed value g, the absolute difference |f - g|
    is guaranteed to be at most 2^minexp.  (Note that it is not possible to
    guarantee error tolerances smaller than machine epsilon relative to the
    largest value within a block.)  This error tolerance is not always tight
    (especially for 3D arrays), but can conservatively be set so that even
    for worst-case inputs the error tolerance is respected.  To achieve
    fixed accuracy to within 'tolerance', use the -a <tolerance> option,
    which sets minexp = floor(log2(tolerance)), minbits = 0, maxbits = 4171,
    and maxprec = 64.  As in fixed-precision mode, the number of bits used
    per block is not fixed but is dictated by the data.  Use -a 0 to achieve
    near-lossless compression.  Fixed-accuracy mode gives the highest quality
    (in terms of absolute error) for a given compression rate, and is
    preferable when random access is not needed.

As mentioned above, other combinations of constraints can be used.
For example, to ensure that the compressed stream is not larger than
the uncompressed one, or that it fits within the amount of memory
allocated, one may in conjunction with other constraints set
maxbits = 4^d * CHAR_BIT * sizeof(Type), where Type is either float or
double.  The minbits parameter is useful only in fixed-rate mode--when
minbits = maxbits, zero-bits are padded to blocks that compress to fewer
than maxbits bits.

The 'speed' program takes two optional parameters:

    speed [rate] [blocks]

It measures the throughput of compression and decompression of 3D
double-precision data (in megabytes of uncompressed data per second).
By default, a rate of 1 bit/value and two million blocks are
processed.

The 'pgm' program illustrates how zfp can be used to compress grayscale
images in the pgm format.  The usage is:

    pgm <param> <input.pgm >output.pgm

If param is positive, it is interpreted as the rate in bits per pixel,
which ensures that each block of 4x4 pixels is compressed to a fixed
number of bits, as in texture compression codecs.  If param is negative,
then fixed-precision mode is used with precision -param, which tends to
give higher quality for the same rate.  This use of zfp is not intended
to compete with existing texture and image compression formats, but
exists merely to demonstrate how to compress 8-bit integer data with zfp.

Finally, the 'testzfp' program performs regression testing that exercises
most of the functionality of libzfp and the array classes.  The tests
assume the default compiler settings, i.e. with none of the macros in
Config defined.  By default, small, pregenerated floating-point arrays are
used in the test, since they tend to have the same binary representation
across platforms, whereas it can be difficult to computationally generate
bit-for-bit identical arrays.  To test larger arrays, modify the TESTZFP_*
macros in Config.  When large arrays are used, the (de)compression
throughput is also measured and reported in number of uncompressed bytes
per second.


LIMITATIONS AND MISSING FEATURES
--------------------------------

zfp is released as a beta version with the intent of giving people access
to the code and soliciting feedback on how to improve zfp for the first
full release.  As such, the zfp API is experimental and has not been
fixed, and it is entirely possible that future versions will employ a
different API or even a different codec.

Below is a list of known limitations and desirable features that may make
it into future versions of zfp.

- The current version of zfp allows for near lossless compression through
  suitable parameter choices, but no guarantees are made that bit-for-bit
  lossless compression is achieved.  We envision supporting lossless
  compression in a future version by compressing the difference between
  the original data and nearly losslessly compressed data.

- Special values like infinity and NaN are not supported.  Denormalized
  floating-point numbers are, however, correctly handled.  There is an
  implicit assumption that floating point conforms to IEEE, though
  extensions to other floating-point formats should be possible with
  minor effort.

- No iterators are provided for traversing an array, and currently one
  has to use integer indexing.  Performance could in cases be improved
  by limiting the traversal to sequential access.

- It is not possible to access subarrays via pointers, e.g. via
  double* p = &a[offset]; p[i] = ...  A pointer proxy class similar to
  the reference class would be useful.

- There currently is no way to make a complete copy of a compressed
  array, i.e. a = b; does not work for arrays a and b.

- zfp can potentially provide higher precision than conventional float
  and double arrays, but the interface currently does not expose this.
  For example, such added precision could be useful in finite difference
  computations, where catastrophic cancellation can be an issue when
  insufficient precision is available.

- Only single and double precision types are supported.  Generalizations
  to IEEE half and quad precision would be useful.  For instance,
  compressed 64-bit-per-value storage of 128-bit quad-precision numbers
  could greatly improve the accuracy of double-precision floating-point
  computations using the same amount of storage.

- zfp arrays are not thread-safe.  We are considering options for
  supporting multi-threaded access, e.g. for OpenMP parallelization.

- This version of zfp does not run on the GPU.  Some work has been done to
  port zfp to CUDA, and we expect to release such a version in the future.


QUESTIONS, COMMENTS, AND BUG REPORTS
------------------------------------

For bug reports, questions, and suggestions for improvements, please
contact Peter Lindstrom at pl@llnl.gov.  If you end up using zfp in an
application, please consider sharing with the author your success story
and/or any issues encountered.
