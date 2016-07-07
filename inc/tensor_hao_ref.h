#ifndef TENSOR_HAO_REF_H
#define TENSOR_HAO_REF_H

#include <iomanip>
#include "tensor_core.h"

namespace tensor_hao
{

 template<class T = double, int D = 1> class Tensor_hao_ref : public Tensor_core<T, D>
 {
  public:
     //============
     //CONSTRUCTORS
     //============

     Tensor_hao_ref(void): Tensor_core<T,D>() 
     { 
         //std::cout<<"In Tensor_hao_ref void constructor "<<std::endl;
     }

     //Variadic template 
     template<typename... Values>
     Tensor_hao_ref(int input, Values... inputs)
     {
         int  len = sizeof...(Values);
         int vals[] = {inputs...};

         if( (len+1) != D) {std::cout<<"Length of inputs number is not consistent with template class!!! "<<len<<" "<<D<<std::endl; exit(1);}

         this->n[0] = input;
         if(len>0) std::copy(vals, vals+len, this->n+1);

         this->n_step[0]=1; for(int i=1; i<D; i++) {this->n_step[i] = (this->n_step[i-1]) * (this->n[i-1]);}

         this->L = this->n_step[D-1] * ( this->n[D-1] );

         this->p = nullptr;
     }

     Tensor_hao_ref(Tensor_hao_ref<T, D>& x)
     {
         copy_ref(x);
         //std::cout<<"In Tensor_hao_ref constructor "<<std::endl;
     }

     Tensor_hao_ref(Tensor_core<T, D>& x)
     {
         copy_ref(x);
         //std::cout<<"In Tensor_core constructor "<<std::endl;
     }

     ~Tensor_hao_ref() {}

     Tensor_hao_ref<T, D> & operator  = (Tensor_hao_ref<T, D>& x) 
     {
         if(&x!=this) copy_ref(x);
         //std::cout<<"In Tensor_hao_ref assginment "<<std::endl;
         return *this;
     }

     Tensor_hao_ref<T, D> & operator  = (Tensor_core<T, D>& x)
     {
         if(&x!=this) copy_ref(x);
         //std::cout<<"In Tensor_core assginment "<<std::endl;
         return *this;
     }

     Tensor_hao_ref<T, D> & operator  = (const std::initializer_list <T> &args)
     {
         if( !(this->p) ) {std::cout<<"Tensor_hao_ref has not point to any memory space yet, can not copy from list!"<<std::endl; exit(1);}
         this->copy_list(args);
         return *this;
     }

     //=========
     //FUNCTIONS
     //=========

     void point(T* p_in) { this->p=p_in; }

     void point(std::vector<T>& vec)
     {
         if( this->L != vec.size() )
         {
             std::cout<<"Size is not consistent between ref matrix and vector.\n";
             std::cout<<this->L<<" "<<vec.size()<<std::endl;
             exit(1);
         }
         this->p=vec.data();
     }

  private:
     void copy_ref(Tensor_core<T, D>& x)
     {
         for(int i=0; i<D; i++)
         {
             this->n[i]=x.rank(i);
             this->n_step[i]=x.rank_step(i);
         }
         this->L = x.size();
         this->p = x.data();
     }

 }; //end class Tensor_hao_ref

} //end namespace tensor_hao

#endif

