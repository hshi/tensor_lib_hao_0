#ifndef TENSOR_2D_COMMON_FUN_H
#define TENSOR_2D_COMMON_FUN_H

#include <vector>
#include "tensor_core.h"
#include "tensor_hao_ref.h"
#include "tensor_hao.h"
#include "tensor_element_wise.h"

namespace tensor_hao
{

 template <class T>
 Tensor_hao<T,2> trans(const Tensor_core<T,2>& A)
 {
     int L0 = A.rank(0); int L1 = A.rank(1);
     Tensor_hao<T,2> B(L1, L0);
     for(int i=0; i<L0; i++)
     {
         for(int j=0; j<L1; j++) B(j,i)=A(i,j);
     }
     return B;
 }

 template <class T>
 Tensor_hao<T,2> conjtrans(const Tensor_core<T,2>& A)
 {
     int L0 = A.rank(0); int L1 = A.rank(1);
     Tensor_hao<T,2> B(L1, L0);
     for(int i=0; i<L0; i++)
     {
         for(int j=0; j<L1; j++) B(j,i)=std::conj( A(i,j) );
     }
     return B;
 }


} //end namespace tensor_hao

#endif
