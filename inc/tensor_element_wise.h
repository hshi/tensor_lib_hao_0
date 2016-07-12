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


 //for add tensor + tensor
 template <class T, int D>
 Tensor_hao<T,D> operator + (const Tensor_hao<T,D>& A,const Tensor_hao<T,D>& B) {Tensor_hao<T,D> C=A; C+=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator + (const Tensor_hao<T,D>& A, Tensor_hao<T,D>&& B)     {Tensor_hao<T,D> C=std::move(B); C+=A; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator + (Tensor_hao<T,D>&& A,const Tensor_hao<T,D>& B)      {Tensor_hao<T,D> C=std::move(A); C+=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator + (Tensor_hao<T,D>&& A,Tensor_hao<T,D>&& B)           {Tensor_hao<T,D> C=std::move(A); C+=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator + (const Tensor_hao<T,D>& A,const Tensor_hao_ref<T,D>& B) {Tensor_hao<T,D> C=A; C+=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator + (Tensor_hao<T,D>&& A,const Tensor_hao_ref<T,D>& B)      {Tensor_hao<T,D> C=std::move(A); C+=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator + (const Tensor_hao_ref<T,D>& A,const Tensor_hao<T,D>& B) {Tensor_hao<T,D> C=A; C+=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator + (const Tensor_hao_ref<T,D>& A, Tensor_hao<T,D>&& B)  {Tensor_hao<T,D> C=std::move(B); C+=A; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator + (const Tensor_hao_ref<T,D>& A,const Tensor_hao_ref<T,D>& B) {Tensor_hao<T,D> C=A; C+=B; return C;}


 //for minus tensor - tensor
 template <class T, int D>
 Tensor_hao<T,D> operator - (const Tensor_hao<T,D>& A,const Tensor_hao<T,D>& B) {Tensor_hao<T,D> C=A; C-=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator - (const Tensor_hao<T,D>& A, Tensor_hao<T,D>&& B) {Tensor_hao<T,D> C=std::move(B); C.min_add_equal(A); return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator - (Tensor_hao<T,D>&& A,const Tensor_hao<T,D>& B)      {Tensor_hao<T,D> C=std::move(A); C-=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator - (Tensor_hao<T,D>&& A,Tensor_hao<T,D>&& B)           {Tensor_hao<T,D> C=std::move(A); C-=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator - (const Tensor_hao<T,D>& A,const Tensor_hao_ref<T,D>& B) {Tensor_hao<T,D> C=A; C-=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator - (Tensor_hao<T,D>&& A,const Tensor_hao_ref<T,D>& B)      {Tensor_hao<T,D> C=std::move(A); C-=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator - (const Tensor_hao_ref<T,D>& A,const Tensor_hao<T,D>& B) {Tensor_hao<T,D> C=A; C-=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator - (const Tensor_hao_ref<T,D>& A, Tensor_hao<T,D>&& B)  {Tensor_hao<T,D> C=std::move(B); C.min_add_equal(A); return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator - (const Tensor_hao_ref<T,D>& A,const Tensor_hao_ref<T,D>& B) {Tensor_hao<T,D> C=A; C-=B; return C;}


 //for time tensor * tensor
 template <class T, int D>
 Tensor_hao<T,D> operator * (const Tensor_hao<T,D>& A,const Tensor_hao<T,D>& B) {Tensor_hao<T,D> C=A; C*=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator * (const Tensor_hao<T,D>& A, Tensor_hao<T,D>&& B)     {Tensor_hao<T,D> C=std::move(B); C*=A; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator * (Tensor_hao<T,D>&& A,const Tensor_hao<T,D>& B)      {Tensor_hao<T,D> C=std::move(A); C*=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator * (Tensor_hao<T,D>&& A,Tensor_hao<T,D>&& B)           {Tensor_hao<T,D> C=std::move(A); C*=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator * (const Tensor_hao<T,D>& A,const Tensor_hao_ref<T,D>& B) {Tensor_hao<T,D> C=A; C*=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator * (Tensor_hao<T,D>&& A,const Tensor_hao_ref<T,D>& B)      {Tensor_hao<T,D> C=std::move(A); C*=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator * (const Tensor_hao_ref<T,D>& A,const Tensor_hao<T,D>& B) {Tensor_hao<T,D> C=A; C*=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator * (const Tensor_hao_ref<T,D>& A, Tensor_hao<T,D>&& B)  {Tensor_hao<T,D> C=std::move(B); C*=A; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator * (const Tensor_hao_ref<T,D>& A,const Tensor_hao_ref<T,D>& B) {Tensor_hao<T,D> C=A; C*=B; return C;}

 //for divide tensor / tensor
 template <class T, int D>
 Tensor_hao<T,D> operator / (const Tensor_hao<T,D>& A,const Tensor_hao<T,D>& B) {Tensor_hao<T,D> C=A; C/=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator / (const Tensor_hao<T,D>& A, Tensor_hao<T,D>&& B) {Tensor_hao<T,D> C=std::move(B); C.inv_div_equal(A); return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator / (Tensor_hao<T,D>&& A,const Tensor_hao<T,D>& B)      {Tensor_hao<T,D> C=std::move(A); C/=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator / (Tensor_hao<T,D>&& A,Tensor_hao<T,D>&& B)           {Tensor_hao<T,D> C=std::move(A); C/=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator / (const Tensor_hao<T,D>& A,const Tensor_hao_ref<T,D>& B) {Tensor_hao<T,D> C=A; C/=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator / (Tensor_hao<T,D>&& A,const Tensor_hao_ref<T,D>& B)      {Tensor_hao<T,D> C=std::move(A); C/=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator / (const Tensor_hao_ref<T,D>& A,const Tensor_hao<T,D>& B) {Tensor_hao<T,D> C=A; C/=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator / (const Tensor_hao_ref<T,D>& A, Tensor_hao<T,D>&& B)  {Tensor_hao<T,D> C=std::move(B); C.inv_div_equal(A); return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator / (const Tensor_hao_ref<T,D>& A,const Tensor_hao_ref<T,D>& B) {Tensor_hao<T,D> C=A; C/=B; return C;}

 //for add: tensor + scalar
 template <class T, int D>
 Tensor_hao<T,D> operator + (const Tensor_hao<T,D>& A,T B) {Tensor_hao<T,D> C=A; C+=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator + (Tensor_hao<T,D>&& A, T B)     {Tensor_hao<T,D> C=std::move(A); C+=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator + (T B,const Tensor_hao<T,D>& A) {Tensor_hao<T,D> C=A; C+=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator + (T B,Tensor_hao<T,D>&& A)      {Tensor_hao<T,D> C=std::move(A); C+=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator + (const Tensor_hao_ref<T,D>& A,T B) {Tensor_hao<T,D> C=A; C+=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator + (T B,const Tensor_hao_ref<T,D>& A) {Tensor_hao<T,D> C=A; C+=B; return C;}

 //for minus: tensor - scalar
 template <class T, int D>
 Tensor_hao<T,D> operator - (const Tensor_hao<T,D>& A,T B) {Tensor_hao<T,D> C=A; C-=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator - (Tensor_hao<T,D>&& A, T B)     {Tensor_hao<T,D> C=std::move(A); C-=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator - (T B,const Tensor_hao<T,D>& A) {Tensor_hao<T,D> C=A; C.min_add_equal(B);return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator - (T B,Tensor_hao<T,D>&& A)      {Tensor_hao<T,D> C=std::move(A);C.min_add_equal(B);return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator - (const Tensor_hao_ref<T,D>& A,T B) {Tensor_hao<T,D> C=A; C-=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator - (T B,const Tensor_hao_ref<T,D>& A) {Tensor_hao<T,D> C=A; C.min_add_equal(B);return C;}

 //for time: tensor * scalar
 template <class T, int D>
 Tensor_hao<T,D> operator * (const Tensor_hao<T,D>& A,T B) {Tensor_hao<T,D> C=A; C*=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator * (Tensor_hao<T,D>&& A, T B)     {Tensor_hao<T,D> C=std::move(A); C*=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator * (T B,const Tensor_hao<T,D>& A) {Tensor_hao<T,D> C=A; C*=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator * (T B,Tensor_hao<T,D>&& A)      {Tensor_hao<T,D> C=std::move(A); C*=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator * (const Tensor_hao_ref<T,D>& A,T B) {Tensor_hao<T,D> C=A; C*=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator * (T B,const Tensor_hao_ref<T,D>& A) {Tensor_hao<T,D> C=A; C*=B; return C;}

 //for minus: tensor / scalar
 template <class T, int D>
 Tensor_hao<T,D> operator / (const Tensor_hao<T,D>& A,T B) {Tensor_hao<T,D> C=A; C/=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator / (Tensor_hao<T,D>&& A, T B)     {Tensor_hao<T,D> C=std::move(A); C/=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator / (T B,const Tensor_hao<T,D>& A) {Tensor_hao<T,D> C=A; C.inv_div_equal(B);return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator / (T B,Tensor_hao<T,D>&& A)      {Tensor_hao<T,D> C=std::move(A);C.inv_div_equal(B);return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator / (const Tensor_hao_ref<T,D>& A,T B) {Tensor_hao<T,D> C=A; C/=B; return C;}
 template <class T, int D>
 Tensor_hao<T,D> operator / (T B,const Tensor_hao_ref<T,D>& A) {Tensor_hao<T,D> C=A; C.inv_div_equal(B);return C;}



} //end namespace tensor_hao

#endif
