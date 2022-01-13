#ifndef ZFP_ARRAY1_H
#define ZFP_ARRAY1_H

#include "zfparray.h"
#include "zfpcodec.h"
#include "cache.h"

namespace zfp {

// compressed 1D array of scalars
template < typename Scalar, class Codec = zfp::codec<Scalar> >
class array1 : public array {
public:
  array1() : array(1, Codec::type) {}

  // constructor of n-sample array using rate bits per value, at least
  // csize bytes of cache, and optionally initialized from flat array p
  array1(uint n, double rate, const Scalar* p = 0, size_t csize = 0) :
    array(1, Codec::type),
    cache(lines(csize, n))
  {
    set_rate(rate);
    resize(n, p == 0);
    if (p)
      set(p);
  }

  // total number of elements in array
  size_t size() const { return size_t(nx); }

  // resize the array (all previously stored data will be lost)
  void resize(uint n, bool clear = true)
  {
    if (n == 0)
      free();
    else {
      nx = n;
      bx = (nx + 3) / 4;
      blocks = bx;
      alloc(clear);

      // precompute block dimensions
      deallocate(shape);
      if (nx & 3u) {
        shape = (uchar*)allocate(blocks);
        uchar* p = shape;
        for (uint i = 0; i < bx; i++)
          *p++ = (i == bx - 1 ? -nx & 3u : 0);
      }
      else
        shape = 0;
    }
  }

  // cache size in number of bytes
  size_t cache_size() const { return cache.size() * sizeof(CacheLine); }

  // set minimum cache size in bytes (array dimensions must be known)
  void set_cache_size(size_t csize)
  {
    flush_cache();
    cache.resize(lines(csize, nx));
  }

  // empty cache without compressing modified cached blocks
  void clear_cache() const { cache.clear(); }

  // flush cache by compressing all modified cached blocks
  void flush_cache() const
  {
    for (typename Cache<CacheLine>::const_iterator p = cache.first(); p; p++) {
      if (p->tag.dirty()) {
        uint b = p->tag.index() - 1;
        encode(b, p->line->a);
      }
      cache.flush(p->line);
    }
  }

  // decompress array and store at p
  void get(Scalar* p) const
  {
    uint b = 0;
    for (uint i = 0; i < bx; i++, p += 4, b++) {
      const CacheLine* line = cache.lookup(b + 1);
      if (line)
        line->get(p, 1, shape ? shape[b] : 0);
      else
        decode(b, p, 1);
    }
  }

  // initialize array by copying and compressing data stored at p
  void set(const Scalar* p)
  {
    uint b = 0;
    for (uint i = 0; i < bx; i++, b++, p += 4)
      encode(b, p, 1);
    cache.clear();
  }

  // reference to a single array value
  class reference {
  public:
    operator Scalar() const { return array->get(i); }
    reference operator=(const reference& r) { array->set(i, r.operator Scalar()); return *this; }
    reference operator=(Scalar val) { array->set(i, val); return *this; }
    reference operator+=(Scalar val) { array->add(i, val); return *this; }
    reference operator-=(Scalar val) { array->sub(i, val); return *this; }
    reference operator*=(Scalar val) { array->mul(i, val); return *this; }
    reference operator/=(Scalar val) { array->div(i, val); return *this; }
  protected:
    friend class array1;
    reference(array1* array, uint i) : array(array), i(i) {}
    array1* array;
    uint i;
  };

  // (i) accessors
  const Scalar& operator()(uint i) const { return get(i); }
  reference operator()(uint i) { return reference(this, i); }

  // flat index accessors
  const Scalar& operator[](uint index) const { return get(index); }
  reference operator[](uint index) { return reference(this, index); }

protected:
  // cache line representing one block of decompressed values
  class CacheLine {
  public:
    friend class array1;
    const Scalar& operator()(uint i) const { return a[index(i)]; }
    Scalar& operator()(uint i) { return a[index(i)]; }
    // copy cache line
    void get(Scalar* p, int sx) const
    {
      const Scalar* q = a;
      for (uint x = 0; x < 4; x++, p += sx, q++)
        *p = *q;
    }
    void get(Scalar* p, int sx, uint shape) const
    {
      if (!shape)
        get(p, sx);
      else {
        // determine block dimensions
        uint nx = 4 - (shape & 3u); shape >>= 2;
        const Scalar* q = a;
        for (uint x = 0; x < nx; x++, p += sx, q++)
          *p = *q;
      }
    }
  protected:
    static uint index(uint i) { return i & 3u; }
    Scalar a[4];
  };

  // inspector
  const Scalar& get(uint i) const
  {
    CacheLine* p = line(i, false);
    return (*p)(i);
  }

  // mutator
  void set(uint i, Scalar val)
  {
    CacheLine* p = line(i, true);
    (*p)(i) = val;
  }

  // in-place updates
  void add(uint i, Scalar val) { (*line(i, true))(i) += val; }
  void sub(uint i, Scalar val) { (*line(i, true))(i) -= val; }
  void mul(uint i, Scalar val) { (*line(i, true))(i) *= val; }
  void div(uint i, Scalar val) { (*line(i, true))(i) /= val; }

  // return cache line for i; may require write-back and fetch
  CacheLine* line(uint i, bool write) const
  {
    CacheLine* p = 0;
    uint b = block(i);
    typename Cache<CacheLine>::Tag t = cache.access(p, b + 1, write);
    uint c = t.index() - 1;
    if (c != b) {
      // write back occupied cache line if it is dirty
      if (t.dirty())
        encode(c, p->a);
      // fetch cache line
      decode(b, p->a);
    }
    return p;
  }

  // encode block with given index
  void encode(uint index, const Scalar* block) const
  {
    stream_wseek(stream->stream, index * blkbits);
    Codec::encode_block_1(stream, block, shape ? shape[index] : 0);
    stream_flush(stream->stream);
  }

  // encode block with given index from strided array
  void encode(uint index, const Scalar* p, int sx) const
  {
    stream_wseek(stream->stream, index * blkbits);
    Codec::encode_block_strided_1(stream, p, shape ? shape[index] : 0, sx);
    stream_flush(stream->stream);
  }

  // decode block with given index
  void decode(uint index, Scalar* block) const
  {
    stream_rseek(stream->stream, index * blkbits);
    Codec::decode_block_1(stream, block, shape ? shape[index] : 0);
  }

  // decode block with given index to strided array
  void decode(uint index, Scalar* p, int sx) const
  {
    stream_rseek(stream->stream, index * blkbits);
    Codec::decode_block_strided_1(stream, p, shape ? shape[index] : 0, sx);
  }

  // block index for i
  static uint block(uint i) { return i / 4; }

  // number of cache lines corresponding to size (or suggested size if zero)
  static uint lines(size_t size, uint n)
  {
    n = uint((size ? size : 8 * sizeof(Scalar)) / sizeof(CacheLine));
    return std::max(n, 1u);
  }

  mutable Cache<CacheLine> cache; // cache of decompressed blocks
};

typedef array1<float> array1f;
typedef array1<double> array1d;

}

#endif
