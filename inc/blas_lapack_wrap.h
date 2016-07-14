#ifndef BLAS_LAPACK_WRAP_H
#define BLAS_LAPACK_WRAP_H

#include<complex>

#ifdef USE_MKL

  #define MKL_Complex8  std::complex<float>
  #define MKL_Complex16 std::complex<double>
  #include "mkl.h"
  #define F77NAME(x) x

#endif


#ifdef USE_ACML

  #define ACML_Complex8  std::complex<float>
  #define ACML_Complex16 std::complex<double>
  #include "acml_customized.h" // change acml.h, see details inside
  #define F77NAME(x) x##_

#endif


#ifdef USE_BLAS_LAPACK

  #define BLAS_LAPACK_Complex8  std::complex<float>
  #define BLAS_LAPACK_Complex16 std::complex<double>
  #include "blas_lapack_customized.h" // write own header file in blas F77NAME(x) is defined inside
#endif

#endif
