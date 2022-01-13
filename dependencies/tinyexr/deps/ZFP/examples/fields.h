#ifndef FIELDS_H
#define FIELDS_H

// single- and double-precision fields for regression testing

extern const float array_float[3][4096];
extern const double array_double[3][4096];

template <typename Scalar>
struct Field {
  static const Scalar (*array)[4096];
};

template <>
const float (*Field<float>::array)[4096] = array_float;

template <>
const double (*Field<double>::array)[4096] = array_double;

#endif
