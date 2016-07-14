#ifndef TENSOR_BL_CPU_H
#define TENSOR_BL_CPU_H

#include <vector>
#include "tensor_core.h"
#include "tensor_hao_ref.h"
#include "tensor_hao.h"
#include "tensor_element_wise.h"
#include "tensor_2d_common_fun.h"

namespace tensor_hao
{

 /*************************************/
 /*Matrix Multiply C=alpha*A.B+beta*C */
 /*************************************/

 void gmm_cpu(const Tensor_core<float,2>& A, const Tensor_core<float,2>& B, Tensor_core<float,2>& C,
          char TRANSA='N', char TRANSB='N', float alpha=1, float beta=0);

 void gmm_cpu(const Tensor_core<double,2>& A, const Tensor_core<double,2>& B, Tensor_core<double,2>& C,
          char TRANSA='N', char TRANSB='N', double alpha=1, double beta=0);

 void gmm_cpu(const Tensor_core<std::complex<float>,2>& A, const Tensor_core<std::complex<float>,2>& B, Tensor_core<std::complex<float>,2>& C,
          char TRANSA='N', char TRANSB='N', std::complex<float> alpha=1, std::complex<float> beta=0);

 void gmm_cpu(const Tensor_core<std::complex<double>,2>& A, const Tensor_core<std::complex<double>,2>& B, Tensor_core<std::complex<double>,2>& C,
          char TRANSA='N', char TRANSB='N', std::complex<double> alpha=1, std::complex<double> beta=0);


 /*****************************************/
 /*Diagonalize symmetric/ Hermitian Matrix*/
 /*****************************************/
 void eigen_cpu(Tensor_core<double,2>& A, Tensor_core<double,1>& W, char JOBZ='V', char UPLO='U');
 void eigen_cpu(Tensor_core<std::complex<double>,2>& A, Tensor_core<double,1>& W, char JOBZ='V', char UPLO='U');


} //end namespace tensor_hao

#endif
