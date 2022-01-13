/* single-precision floating-point traits */

#define Scalar float       /* floating-point type */
#define Int int32          /* corresponding signed integer type */
#define UInt uint32        /* corresponding unsigned integer type */
#define EBITS 8            /* number of exponent bits */
#define NBMASK 0xaaaaaaaau /* negabinary mask */

#if __STDC_VERSION__ >= 199901L
  #define FABS(x)     fabsf(x)
  #define FREXP(x, e) frexpf(x, e)
  #define LDEXP(x, e) ldexpf(x, e)
#else
  #define FABS(x)     (float)fabs(x)
  #define FREXP(x, e) (void)frexp(x, e)
  #define LDEXP(x, e) (float)ldexp(x, e)
#endif
