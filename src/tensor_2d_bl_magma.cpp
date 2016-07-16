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

 /******************************************/
 /*LU Decomposition a complex square Matrix*/
 /******************************************/

 LUDecomp<complex<double>> LUconstruct_magma(const Tensor_core<complex<double>,2>& x)
 {
     if( x.rank(0) != x.rank(1) ) {cout<<"Input for LU is not square matrix!\n"; exit(1);}

     //Create LU object
     LUDecomp<complex<double>> y;
     y.A    = Tensor_hao< complex<double>, 2 > ( x.n_ptr() );
     y.ipiv = Tensor_hao<int,1>( x.rank(0) );

     //Prepare for zgetrf
     magma_int_t M = x.rank(0), N = x.rank(1);
     magma_int_t LDA = ((M+31)/32)*32; 
     magmaDoubleComplex_ptr d_A;  magma_zmalloc(&d_A, LDA*N);
     magma_int_t info;

     //Transfer data and call zgetrf
     magma_zsetmatrix(M, N, (magmaDoubleComplex* ) x.data(), M, d_A, LDA );
     magma_zgetrf_gpu(M, N, d_A, LDA, (magma_int_t*) y.ipiv.data(), &info);
     magma_zgetmatrix(M, N, d_A, LDA, (magmaDoubleComplex* ) y.A.data(), M);
     y.info=info;

     //Clean
     magma_free(d_A);

     if(y.info<0) {cout<<"The "<<y.info<<"-th parameter is illegal in LUconstruct_magma!"<<endl; exit(1);}
     return y;
 }


 /********************************************************************************************************************/
 /*Inverse of matrix: If determinant of the matrix is outof machine precision, inverse should be fine, since it solve*
  *The linear equation, every small value is well defined                                                            */
 /********************************************************************************************************************/
 Tensor_hao<complex<double>,2> inverse_magma(const LUDecomp<complex<double>>& x)
 {
     magma_int_t N=x.A.rank(0); magma_int_t info;

     magmaDoubleComplex_ptr d_A , dwork;
     magma_int_t lda, ldwork;
     lda = ((N+31)/32)*32;             //round up to multiple of 32 for best GPU performance
     ldwork = N*magma_get_zgetri_nb(N); // magma_get_zgetri_nb optimizes the blocksize
     magma_zmalloc( &d_A, lda*N ); magma_zmalloc( &dwork, ldwork );

     //copy matrix from CPU to GPU
     magma_zsetmatrix( N, N, (magmaDoubleComplex* )x.A.data(), N, d_A, lda );

     //calculate the inverse matrix with zgetri
     magma_zgetri_gpu( N, d_A, lda, (magma_int_t*) x.ipiv.data(), dwork, ldwork, &info );
     if(info<0) {cout<<"The "<<info<<"-th parameter is illegal in inverse_magma!"<<endl; exit(1);}

     //copy matrix from GPU to CPU
     Tensor_hao<complex<double>,2> A(N,N);
     magma_zgetmatrix( N, N, d_A, lda, (magmaDoubleComplex* )A.data(), N );

     magma_free(d_A); magma_free(dwork);

     return A;
 }


 /*********************************************************/
 /*Solve linear equation of matrix A*M=B: return M=A^{-1}B*/
 /*********************************************************/
 Tensor_hao<complex<double>,2> solve_lineq_magma(const LUDecomp<complex<double>>& x, const Tensor_core<complex<double>,2>& B, char TRANS)
 {
     if( x.A.rank(0) != B.rank(0) )  {cout<<"Input size for solving linear equation is not consistent!\n"; exit(1);}
     magma_int_t N=B.rank(0); magma_int_t NRHS=B.rank(1); magma_int_t info;

     magma_trans_t Trans = magma_trans_const(TRANS);
     magmaDoubleComplex_ptr d_A, d_B;
     magma_int_t lda, ldb;
     lda = ((N+31)/32)*32;
     ldb = ((N+31)/32)*32;

     //allocate memory on GPU
     magma_zmalloc( &d_A, lda*N );
     magma_zmalloc( &d_B, ldb*NRHS );

     //copy matrix from CPU to GPU
     magma_zsetmatrix( N, N,    (magmaDoubleComplex* )x.A.data(), N, d_A, lda );
     magma_zsetmatrix( N, NRHS, (magmaDoubleComplex* )B.data(),   N, d_B, ldb );

     //Solve the equation
     magma_zgetrs_gpu( Trans, N, NRHS, d_A, lda, (magma_int_t*)x.ipiv.data(), d_B, ldb, &info );
     if(info!=0)
     {
         cout<<"Solve linear equation is not suceesful: "<<info<<"-th parameter is illegal! \n";
         exit(1);
     }

     //copy matrix from GPU to CPU
     Tensor_hao<complex<double>,2> M(N,NRHS);
     magma_zgetmatrix( N, NRHS, d_B, ldb, (magmaDoubleComplex* ) M.data(), N );

     //free memory
     magma_free( d_A );  magma_free( d_B );

     return M;
 }

} //end namespace tensor_hao

#endif
