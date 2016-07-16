#ifdef USE_MAGMA

#include "tensor_2d_bl_magma.h"

using namespace std;

namespace tensor_hao
{

 static magmaFloatComplex  _cast_C(const complex<float>&  Z)  { return MAGMA_C_MAKE(real(Z), imag(Z)); }
 static magmaDoubleComplex _cast_Z(const complex<double>& Z)  { return MAGMA_Z_MAKE(real(Z), imag(Z)); }

 /*************************************/
 /*Matrix Multiply C=alpha*A.B+beta*C */
 /*************************************/
 //Magma_*gemm only support GPU interface.

 void gmm_magma(const Tensor_core<float,2>& A, const Tensor_core<float,2>& B, Tensor_core<float,2>& C,
          char TRANSA, char TRANSB, float alpha, float beta)
 {
     int AL0 = A.rank(0); int AL1 = A.rank(1);
     int BL0 = B.rank(0); int BL1 = B.rank(1);
     int CL0 = C.rank(0); int CL1 = C.rank(1);

     magma_int_t M, N, K, LDA, LDB, LDC;
     magma_trans_t transA=magma_trans_const(TRANSA), transB=magma_trans_const(TRANSB);
     magmaFloat_ptr d_A, d_B, d_C;

     //Set LDA, LDB, and LDC, round up to multiple of 32 for best GPU performance
     LDA = ((AL0+31)/32)*32; LDB = ((BL0+31)/32)*32; LDC = ((CL0+31)/32)*32;

     // Allocate memory for the matrices on GPU 
     magma_smalloc(&d_A, LDA*AL1 );
     magma_smalloc(&d_B, LDB*BL1 );
     magma_smalloc(&d_C, LDC*CL1 );

     // Copy data from host (CPU) to device (GPU)
     magma_ssetmatrix( AL0, AL1, A.data(), AL0, d_A, LDA );
     magma_ssetmatrix( BL0, BL1, B.data(), BL0, d_B, LDB );
     if( abs(beta)>1e-32 ) magma_ssetmatrix( CL0, CL1, C.data(), CL0, d_C, LDC );

     //Call magma_sgemm
     M=( TRANSA=='N' || TRANSA=='n' ) ? AL0:AL1;
     K=( TRANSA=='N' || TRANSA=='n' ) ? AL1:AL0;
     N=( TRANSB=='N' || TRANSB=='n' ) ? BL1:BL0;
     magma_sgemm(transA, transB, M, N, K, alpha, d_A, LDA, d_B, LDB, beta,d_C, LDC);

     // Copy solution from device (GPU) to host (CPU)
     magma_sgetmatrix(CL0, CL1, d_C, LDC, C.data(), CL0);

     // Free memory on GPU
     magma_free(d_A); magma_free(d_B); magma_free(d_C);
 }

 void gmm_magma(const Tensor_core<double,2>& A, const Tensor_core<double,2>& B, Tensor_core<double,2>& C,
          char TRANSA, char TRANSB, double alpha, double beta)
 {
     int AL0 = A.rank(0); int AL1 = A.rank(1);
     int BL0 = B.rank(0); int BL1 = B.rank(1);
     int CL0 = C.rank(0); int CL1 = C.rank(1);

     magma_int_t M, N, K, LDA, LDB, LDC;
     magma_trans_t transA=magma_trans_const(TRANSA), transB=magma_trans_const(TRANSB);
     magmaDouble_ptr d_A, d_B, d_C;

     //Set LDA, LDB, and LDC, round up to multiple of 32 for best GPU performance
     LDA = ((AL0+31)/32)*32; LDB = ((BL0+31)/32)*32; LDC = ((CL0+31)/32)*32;

     // Allocate memory for the matrices on GPU 
     magma_dmalloc(&d_A, LDA*AL1 );
     magma_dmalloc(&d_B, LDB*BL1 );
     magma_dmalloc(&d_C, LDC*CL1 );

     // Copy data from host (CPU) to device (GPU)
     magma_dsetmatrix( AL0, AL1, A.data(), AL0, d_A, LDA );
     magma_dsetmatrix( BL0, BL1, B.data(), BL0, d_B, LDB );
     if( abs(beta)>1e-32 ) magma_dsetmatrix( CL0, CL1, C.data(), CL0, d_C, LDC );

     //Call magma_sgemm
     M=( TRANSA=='N' || TRANSA=='n' ) ? AL0:AL1;
     K=( TRANSA=='N' || TRANSA=='n' ) ? AL1:AL0;
     N=( TRANSB=='N' || TRANSB=='n' ) ? BL1:BL0;
     magma_dgemm(transA, transB, M, N, K, alpha, d_A, LDA, d_B, LDB, beta,d_C, LDC);

     // Copy solution from device (GPU) to host (CPU)
     magma_dgetmatrix(CL0, CL1, d_C, LDC, C.data(), CL0);

     // Free memory on GPU
     magma_free(d_A); magma_free(d_B); magma_free(d_C);
 }

 void gmm_magma(const Tensor_core<complex<float>,2>& A, const Tensor_core<complex<float>,2>& B, Tensor_core<complex<float>,2>& C,
          char TRANSA, char TRANSB, complex<float> alpha, complex<float> beta)
 {
     int AL0 = A.rank(0); int AL1 = A.rank(1);
     int BL0 = B.rank(0); int BL1 = B.rank(1);
     int CL0 = C.rank(0); int CL1 = C.rank(1);

     magma_int_t M, N, K, LDA, LDB, LDC;
     magma_trans_t transA=magma_trans_const(TRANSA), transB=magma_trans_const(TRANSB);
     magmaFloatComplex_ptr d_A, d_B, d_C;

     //Set LDA, LDB, and LDC, round up to multiple of 32 for best GPU performance
     LDA = ((AL0+31)/32)*32; LDB = ((BL0+31)/32)*32; LDC = ((CL0+31)/32)*32;

     // Allocate memory for the matrices on GPU 
     magma_cmalloc(&d_A, LDA*AL1 );
     magma_cmalloc(&d_B, LDB*BL1 );
     magma_cmalloc(&d_C, LDC*CL1 );

     // Copy data from host (CPU) to device (GPU)
     magma_csetmatrix( AL0, AL1, (magmaFloatComplex* ) A.data(), AL0, d_A, LDA );
     magma_csetmatrix( BL0, BL1, (magmaFloatComplex* ) B.data(), BL0, d_B, LDB );
     if( abs(beta)>1e-32 ) magma_csetmatrix( CL0, CL1, (magmaFloatComplex* ) C.data(), CL0, d_C, LDC );

     //Call magma_sgemm
     M=( TRANSA=='N' || TRANSA=='n' ) ? AL0:AL1;
     K=( TRANSA=='N' || TRANSA=='n' ) ? AL1:AL0;
     N=( TRANSB=='N' || TRANSB=='n' ) ? BL1:BL0;
     magma_cgemm(transA, transB, M, N, K, _cast_C(alpha), d_A, LDA, d_B, LDB, _cast_C(beta),d_C, LDC);

     // Copy solution from device (GPU) to host (CPU)
     magma_cgetmatrix(CL0, CL1, d_C, LDC, (magmaFloatComplex* ) C.data(), CL0);

     // Free memory on GPU
     magma_free(d_A); magma_free(d_B); magma_free(d_C);
 }

 void gmm_magma(const Tensor_core<complex<double>,2>& A, const Tensor_core<complex<double>,2>& B, Tensor_core<complex<double>,2>& C,
          char TRANSA, char TRANSB, complex<double> alpha, complex<double> beta)
 {
     int AL0 = A.rank(0); int AL1 = A.rank(1);
     int BL0 = B.rank(0); int BL1 = B.rank(1);
     int CL0 = C.rank(0); int CL1 = C.rank(1);

     magma_int_t M, N, K, LDA, LDB, LDC;
     magma_trans_t transA=magma_trans_const(TRANSA), transB=magma_trans_const(TRANSB);
     magmaDoubleComplex_ptr d_A, d_B, d_C;

     //Set LDA, LDB, and LDC, round up to multiple of 32 for best GPU performance
     LDA = ((AL0+31)/32)*32; LDB = ((BL0+31)/32)*32; LDC = ((CL0+31)/32)*32;

     // Allocate memory for the matrices on GPU 
     magma_zmalloc(&d_A, LDA*AL1 );
     magma_zmalloc(&d_B, LDB*BL1 );
     magma_zmalloc(&d_C, LDC*CL1 );

     // Copy data from host (CPU) to device (GPU)
     magma_zsetmatrix( AL0, AL1, (magmaDoubleComplex* ) A.data(), AL0, d_A, LDA );
     magma_zsetmatrix( BL0, BL1, (magmaDoubleComplex* ) B.data(), BL0, d_B, LDB );
     if( abs(beta)>1e-32 ) magma_zsetmatrix( CL0, CL1, (magmaDoubleComplex* ) C.data(), CL0, d_C, LDC );

     //Call magma_sgemm
     M=( TRANSA=='N' || TRANSA=='n' ) ? AL0:AL1;
     K=( TRANSA=='N' || TRANSA=='n' ) ? AL1:AL0;
     N=( TRANSB=='N' || TRANSB=='n' ) ? BL1:BL0;
     magma_zgemm(transA, transB, M, N, K, _cast_Z(alpha), d_A, LDA, d_B, LDB, _cast_Z(beta),d_C, LDC);

     // Copy solution from device (GPU) to host (CPU)
     magma_zgetmatrix(CL0, CL1, d_C, LDC, (magmaDoubleComplex* ) C.data(), CL0);

     // Free memory on GPU
     magma_free(d_A); magma_free(d_B); magma_free(d_C);
 }

 /******************************/
 /*Diagonalize symmetric Matrix*/
 /******************************/
 void eigen_magma(Tensor_core<double,2>& A, Tensor_core<double,1>& W, char JOBZ, char UPLO)
 {
     if( A.rank(0) != A.rank(1) ) {cout<<"Input for eigen is not square matrix!"<<endl; exit(1);}
     if( A.rank(0) != W.rank(0) ) {cout<<"Input size of W is not consistent with A!"<<endl; exit(1);}

     magma_vec_t jobz = magma_vec_const(JOBZ); magma_uplo_t uplo = magma_uplo_const(UPLO);
     magma_int_t N=A.rank(0); magma_int_t info;

     double work_test[1]; magma_int_t iwork_test[1];
     magma_int_t lwork=-1; magma_int_t liwork=-1;
     magma_dsyevd( jobz, uplo, N, A.data(), N, W.data(), work_test, lwork, iwork_test, liwork, &info );

     lwork=lround(work_test[0]); liwork=iwork_test[0];
     double* work; magma_int_t* iwork;
     magma_dmalloc_cpu(&work, lwork); magma_imalloc_cpu(&iwork, liwork);
     magma_dsyevd( jobz, uplo, N, A.data(), N, W.data(), work, lwork, iwork, liwork, &info );
     magma_free_cpu(work); magma_free_cpu(iwork);

     if(info!=0) {cout<<"Dsyevd failed: info= "<< info<<"\n"; exit(1);}
 }

 /******************************/
 /*Diagonalize Hermition Matrix*/
 /******************************/
 void eigen_magma(Tensor_core<complex<double>,2>& A, Tensor_core<double,1>& W, char JOBZ, char UPLO)
 {
     if( A.rank(0) != A.rank(1) ) {cout<<"Input for eigen is not square matrix!\n"; exit(1);}
     if( A.rank(0) != W.rank(0) ) {cout<<"Input size of W is not consistent with A!"<<endl; exit(1);}

     magma_vec_t jobz = magma_vec_const(JOBZ); magma_uplo_t uplo = magma_uplo_const(UPLO);
     magma_int_t N=A.rank(0); magma_int_t info;

     magmaDoubleComplex work_test[1]; double rwork_test[1]; magma_int_t iwork_test[1];
     magma_int_t lwork=-1; magma_int_t lrwork=-1; magma_int_t liwork=-1;
     magma_zheevd( jobz, uplo, N, (magmaDoubleComplex* ) A.data(), N, W.data(),
                   work_test, lwork, rwork_test, lrwork, iwork_test, liwork, &info );

     lwork=lround( MAGMA_Z_REAL(work_test[0]) ); lrwork=lround(rwork_test[0]); liwork=iwork_test[0];
     magmaDoubleComplex* work; double* rwork; magma_int_t* iwork;
     magma_zmalloc_cpu(&work, lwork); magma_dmalloc_cpu(&rwork, lrwork); magma_imalloc_cpu(&iwork, liwork);
     magma_zheevd( jobz, uplo, N, (magmaDoubleComplex* ) A.data(), N, W.data(),
                   work, lwork, rwork, lrwork, iwork, liwork, &info );

     magma_free_cpu(work); magma_free_cpu(rwork); magma_free_cpu(iwork);
     if(info!=0) {cout<<"Zheevd failed: info= "<< info<<"\n"; exit(1);}
 }


} //end namespace tensor_hao

#endif
