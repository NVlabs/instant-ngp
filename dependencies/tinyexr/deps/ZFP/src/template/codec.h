#define PERM _t1(perm, DIMS)           /* coefficient order */
#define BLOCK_SIZE (1 << (2 * DIMS))   /* values per block */
#define EBIAS ((1 << (EBITS - 1)) - 1) /* exponent bias */
