#ifndef TENSOR_ELEMENT_WISE_H
#define TENSOR_ELEMENT_WISE_H

#include <complex>
#include "tensor_hao_ref.h"
#include "tensor_hao.h"

namespace tensor_hao
{

 template <class T, int D>
 Tensor_hao<std::complex<T>, D> conj(const Tensor_core<std::complex<T>, D>& A) 
 {
     Tensor_hao<std::complex<T>, D> B ( A.n_ptr() );
     int L = A.size(); 
     const std::complex<T>* A_p = A.data(); 
           std::complex<T>* B_p = B.data();
     for(int i=0; i<L; i++) B_p[i] = std::conj( A_p[i] );
     return B;
 }

 template <class T, int D>
 Tensor_hao<T, D> exp(const Tensor_core<T, D>& A)
 {
     Tensor_hao<T, D> B ( A.n_ptr() );
     int L = A.size();
     const T * A_p = A.data();
           T * B_p = B.data();
     for(int i=0; i<L; i++) B_p[i] = std::exp( A_p[i] );
     return B;
 }

 template <class T, int D>
 Tensor_hao<T, D> operator - (const Tensor_core<T, D>& A)
 {
     Tensor_hao<T, D> B ( A.n_ptr() );
     int L = A.size();
     const T * A_p = A.data();
           T * B_p = B.data();
     for(int i=0; i<L; i++) B_p[i] = -A_p[i];
     return B;
 }


} //end namespace tensor_hao

#endif
