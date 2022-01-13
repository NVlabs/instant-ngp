/* double-precision floating-point traits */

#define Scalar double                /* floating-point type */
#define Int int64                    /* corresponding signed integer type */
#define UInt uint64                  /* corresponding unsigned integer type */
#define EBITS 11                     /* number of exponent bits */
#define NBMASK 0xaaaaaaaaaaaaaaaaull /* negabinary mask */

#define FABS(x) fabs(x)
#define FREXP(x, e) frexp(x, e)
#define LDEXP(x, e) ldexp(x, e)
