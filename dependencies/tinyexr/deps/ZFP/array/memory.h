#ifndef MEMORY_H
#define MEMORY_H

#include <cstdlib>
#include "types.h"

inline void*
allocate(size_t size, size_t alignment = 0)
{
#if defined(__USE_XOPEN2K) && defined(ALIGNED_ALLOC)
  void* ptr;
  if (alignment > 1)
    posix_memalign(&ptr, alignment, size);
  else
    ptr = malloc(size);
  return ptr;
#else
  return new uchar[size];
#endif
}

template <typename T>
inline void
deallocate(T* ptr)
{
#if defined(__USE_XOPEN2K) && defined(ALIGNED_ALLOC)
  if (ptr)
    free(ptr);
#else
  delete[] ptr;
#endif
}

template <typename T>
inline void
reallocate(T*& ptr, size_t size, size_t alignment = 0)
{
  deallocate(ptr);
  ptr = static_cast<T*>(allocate(size, alignment));
}

#endif
