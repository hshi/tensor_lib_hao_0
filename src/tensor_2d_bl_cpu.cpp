#include "tensor_2d_bl_cpu.h"
#include "blas_lapack_wrap.h"

using namespace std;

namespace tensor_hao
{

 /*************************************/
 /*Matrix Multiply C=alpha*A.B+beta*C */
 /*************************************/

 void gmm_cpu(const Tensor_core<float,2>& A, const Tensor_core<float,2>& B, Tensor_core<float,2>& C,
          char TRANSA, char TRANSB, float alpha, float beta)
 {
     int  M, N, K, LDA, LDB, LDC;
     M=(TRANSA=='N' || TRANSA=='n' ) ? A.rank(0):A.rank(1);
     K=(TRANSA=='N' || TRANSA=='n' ) ? A.rank(1):A.rank(0);
     N=(TRANSB=='N' || TRANSB=='n' ) ? B.rank(1):B.rank(0);
     LDA=A.rank(0);
     LDB=B.rank(0);
     LDC=C.rank(0);

     F77NAME(sgemm)(&TRANSA, &TRANSB, &M, &N, &K, &alpha, A.data(), &LDA, B.data(), &LDB, &beta, C.data(), &LDC);
 }

 void gmm_cpu(const Tensor_core<double,2>& A, const Tensor_core<double,2>& B, Tensor_core<double,2>& C,
          char TRANSA, char TRANSB, double alpha, double beta)
 {
     int  M, N, K, LDA, LDB, LDC;
     M=(TRANSA=='N' || TRANSA=='n' ) ? A.rank(0):A.rank(1);
     K=(TRANSA=='N' || TRANSA=='n' ) ? A.rank(1):A.rank(0);
     N=(TRANSB=='N' || TRANSB=='n' ) ? B.rank(1):B.rank(0);
     LDA=A.rank(0);
     LDB=B.rank(0);
     LDC=C.rank(0);

     F77NAME(dgemm)(&TRANSA, &TRANSB, &M, &N, &K, &alpha, A.data(), &LDA, B.data(), &LDB, &beta, C.data(), &LDC);
 }

 void gmm_cpu(const Tensor_core<std::complex<float>,2>& A, const Tensor_core<std::complex<float>,2>& B, Tensor_core<std::complex<float>,2>& C,
          char TRANSA, char TRANSB, std::complex<float> alpha, std::complex<float> beta)
 {
     int  M, N, K, LDA, LDB, LDC;
     M=(TRANSA=='N' || TRANSA=='n' ) ? A.rank(0):A.rank(1);
     K=(TRANSA=='N' || TRANSA=='n' ) ? A.rank(1):A.rank(0);
     N=(TRANSB=='N' || TRANSB=='n' ) ? B.rank(1):B.rank(0);
     LDA=A.rank(0);
     LDB=B.rank(0);
     LDC=C.rank(0);

     F77NAME(cgemm)(&TRANSA, &TRANSB, &M, &N, &K, &alpha, A.data(), &LDA, B.data(), &LDB, &beta, C.data(), &LDC);
 }

 void gmm_cpu(const Tensor_core<std::complex<double>,2>& A, const Tensor_core<std::complex<double>,2>& B, Tensor_core<std::complex<double>,2>& C,
          char TRANSA, char TRANSB, std::complex<double> alpha, std::complex<double> beta)
 {
     int  M, N, K, LDA, LDB, LDC;
     M=(TRANSA=='N' || TRANSA=='n' ) ? A.rank(0):A.rank(1);
     K=(TRANSA=='N' || TRANSA=='n' ) ? A.rank(1):A.rank(0);
     N=(TRANSB=='N' || TRANSB=='n' ) ? B.rank(1):B.rank(0);
     LDA=A.rank(0);
     LDB=B.rank(0);
     LDC=C.rank(0);

     F77NAME(zgemm)(&TRANSA, &TRANSB, &M, &N, &K, &alpha, A.data(), &LDA, B.data(), &LDB, &beta, C.data(), &LDC);
 }


 /******************************/
 /*Diagonalize symmetric Matrix*/
 /******************************/
 void eigen_cpu(Tensor_core<double,2>& A, Tensor_core<double,1>& W, char JOBZ, char UPLO)
 {
     if( A.rank(0) != A.rank(1) ) {cout<<"Input for eigen is not square matrix!\n"; exit(1);}
     int N=A.rank(0); int info;

     double work_test[1]; int iwork_test[1]; int lwork=-1; int liwork=-1;
     F77NAME(dsyevd)(&JOBZ, &UPLO, &N, A.data(), &N, W.data(), work_test, &lwork, iwork_test, &liwork ,&info);

     lwork=lround(work_test[0]); liwork=iwork_test[0];
     vector<double> work(lwork); vector<int> iwork(liwork);
     F77NAME(dsyevd)(&JOBZ, &UPLO, &N, A.data(), &N, W.data(), work.data(), &lwork, iwork.data(), &liwork ,&info);

     if(info!=0) {cout<<"Dsyevd failed: info= "<< info<<"\n"; exit(1);}
 }


 /******************************/
 /*Diagonalize Hermition Matrix*/
 /******************************/
 void eigen_cpu(Tensor_core<complex<double>,2>& A, Tensor_core<double,1>& W, char JOBZ, char UPLO)
 {
     if( A.rank(0) != A.rank(1) ) {cout<<"Input for eigen is not square matrix!\n"; exit(1);}
     int N=A.rank(0); int info;

     complex<double> work_test[1]; double rwork_test[1]; int iwork_test[1];
     int lwork=-1; int lrwork=-1; int liwork=-1;
     F77NAME(zheevd)(&JOBZ, &UPLO, &N, A.data(), &N, W.data(), work_test, &lwork, rwork_test, &lrwork, iwork_test, &liwork ,&info);

     lwork=lround(work_test[0].real()); lrwork=lround(rwork_test[0]); liwork=iwork_test[0];
     vector<complex<double>> work(lwork); vector<double> rwork(lrwork); vector<int> iwork(liwork);
     F77NAME(zheevd)(&JOBZ, &UPLO, &N, A.data(), &N, W.data(), work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork ,&info);

     if(info!=0) {cout<<"Zheevd failed: info= "<< info<<"\n"; exit(1);}
 }

 /******************************************/
 /*LU Decomposition a complex square Matrix*/
 /******************************************/
 LUDecomp<complex<double>> LUconstruct_cpu(const Tensor_core<complex<double>,2>& x)
 {
     if( x.rank(0) != x.rank(1) ) {cout<<"Input for LU is not square matrix!\n"; exit(1);}
     int N=x.rank(0);
     LUDecomp<complex<double>> y; y.A=x; y.ipiv=Tensor_hao<int,1>(N);

     F77NAME(zgetrf)(&N, &N, y.A.data(), &N, y.ipiv.data(), &(y.info) );
     if(y.info<0) {cout<<"The "<<y.info<<"-th parameter is illegal!\n"; exit(1);}
     return y;
 }



} //end namespace tensor_hao
