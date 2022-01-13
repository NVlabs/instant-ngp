#ifndef INLINE_H
#define INLINE_H

#ifndef _inline
  #if __STDC_VERSION__ >= 199901L
    #define _inline static inline
  #else
    #define _inline static
  #endif
#endif

#endif
