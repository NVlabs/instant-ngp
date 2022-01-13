/* order coefficients by polynomial degree/frequency */
_cache_align(static const uchar perm_1[4]) = {
  0, 1, 2, 3
};

/* maximum number of bit planes to encode */
static uint
_t2(precision, Scalar, 1)(int maxexp, uint maxprec, int minexp)
{
  return MIN(maxprec, MAX(0, maxexp - minexp + 4));
}
