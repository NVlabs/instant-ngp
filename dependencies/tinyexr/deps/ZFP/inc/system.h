#ifndef SYSTEM_H
#define SYSTEM_H

#if __STDC_VERSION__ >= 199901L
  #define _restrict restrict
#else
  #define _restrict
#endif

#ifdef __GNUC__
  #ifndef CACHE_LINE_SIZE
    #define CACHE_LINE_SIZE 0x100
  #endif
  #define _align(n) __attribute__((aligned(n)))
  #define _cache_align(x) x _align(CACHE_LINE_SIZE)
#else
  #define _cache_align(x) x
#endif

#endif
