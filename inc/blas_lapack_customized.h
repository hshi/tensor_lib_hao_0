#ifndef BLAS_LAPACK_CUSTOMIZED_H
#define BLAS_LAPACK_CUSTOMIZED_H

#ifdef FORTRAN_NO_TRAILING_UNDERSCORE
#define F77NAME(x) x
#else
#define F77NAME(x) x##_
#endif

#ifndef BLAS_LAPACK_Complex8
typedef struct
{
  float real, imag;
} BLAS_LAPACK_Complex8;
#endif

#ifndef BLAS_LAPACK_Complex16
typedef struct
{
  double real, imag;
} BLAS_LAPACK_Complex16;
#endif



#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void F77NAME(sgemm)(const char *transa, const char *transb, const int *m, const int *n, const int *k,
           const float *alpha, const float *a, const int *lda, const float *b, const int *ldb,
           const float *beta, float *c, const int *ldc);
void F77NAME(dgemm)(const char *transa, const char *transb, const int *m, const int *n, const int *k,
           const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
           const double *beta, double *c, const int *ldc);
void F77NAME(cgemm)(const char *transa, const char *transb, const int *m, const int *n, const int *k,
           const BLAS_LAPACK_Complex8 *alpha, const BLAS_LAPACK_Complex8 *a, const int *lda,
           const BLAS_LAPACK_Complex8 *b, const int *ldb, const BLAS_LAPACK_Complex8 *beta,
           BLAS_LAPACK_Complex8 *c, const int *ldc);
void F77NAME(zgemm)(const char *transa, const char *transb, const int *m, const int *n, const int *k,
           const BLAS_LAPACK_Complex16 *alpha, const BLAS_LAPACK_Complex16 *a, const int *lda,
           const BLAS_LAPACK_Complex16 *b, const int *ldb, const BLAS_LAPACK_Complex16 *beta,
           BLAS_LAPACK_Complex16 *c, const int *ldc);

void F77NAME(dsyevd)( const char* jobz, const char* uplo, const int* n, double* a,
             const int* lda, double* w, double* work, const int* lwork,
             int* iwork, const int* liwork, int* info );
void F77NAME(zheev)( const char* jobz, const char* uplo, const int* n,
            BLAS_LAPACK_Complex16* a, const int* lda, double* w,
            BLAS_LAPACK_Complex16* work, const int* lwork, double* rwork,
            int* info );
void F77NAME(zheevd)( const char* jobz, const char* uplo, const int* n,
             BLAS_LAPACK_Complex16* a, const int* lda, double* w,
             BLAS_LAPACK_Complex16* work, const int* lwork, double* rwork,
             const int* lrwork, int* iwork, const int* liwork,
             int* info );
void F77NAME(zgetrf)( const int* m, const int* n, BLAS_LAPACK_Complex16* a,
             const int* lda, int* ipiv, int* info );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
