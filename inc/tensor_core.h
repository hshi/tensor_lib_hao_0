#ifndef TENSOR_CORE_H
#define TENSOR_CORE_H

#include <cstdarg>
#include "tensor_base.h"

namespace tensor_hao
{

 template<class T = double, int D =1 > class Tensor_core : public Tensor_base<T>
 {
  protected:
     int n[D];
     int n_step[D];

     Tensor_core(void):Tensor_base<T>() {}
     ~Tensor_core() {}

  public:

     inline const int rank(int i) const 
     {
         #ifndef NDEBUG
         if( i >= D || i<0 ) { std::cout<<"Input i for rank() should be [0, D)!!! "<<i<<" "<<D<<std::endl; exit(1); }
         #endif

         return n[i];
     }


     inline const int rank_step(int i) const 
     {
         #ifndef NDEBUG
         if( i >= D || i<0 ) { std::cout<<"Input i for rank_step() should be [0, D)!!! "<<i<<" "<<D<<std::endl; exit(1); }
         #endif

         return n_step[i];
     }


     //Read elements: for D=1
     inline T operator () (int i0) const 
     {
         #ifndef NDEBUG
         if( D != 1    )         { std::cout<<"Tensor_core::operator(int) only works for D=1 !!!"<<std::endl; exit(1); }
         if(i0 >= n[0] || i0<0 ) { std::cout<<"i0 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         #endif 
         return this->p[ i0 ];
     }

     inline T& operator () (int i0)
     {
         #ifndef NDEBUG
         if( D != 1    )         { std::cout<<"Tensor_core::operator(int) only works for D=1 !!!"<<std::endl; exit(1); }
         if(i0 >= n[0] || i0<0 ) { std::cout<<"i0 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         #endif
         return this->p[ i0 ];
     }


     //Read elements: for D=2
     inline T operator () (int i0, int i1) const 
     {
         #ifndef NDEBUG
         if(D != 2) { std::cout<<"Tensor_core::operator(int, int) only works for D=2 !!!"<<std::endl; exit(1); }
         if(i0 >= n[0] || i0<0 ) { std::cout<<"i0 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); } 
         if(i1 >= n[1] || i1<0 ) { std::cout<<"i1 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         #endif

         return this->p[ i0+i1*n_step[1] ]; 
     }

     inline T& operator () (int i0, int i1)
     {
         #ifndef NDEBUG
         if(D != 2) { std::cout<<"Tensor_core::operator(int, int) only works for D=2 !!!"<<std::endl; exit(1); }
         if(i0 >= n[0] || i0<0 ) { std::cout<<"i0 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         if(i1 >= n[1] || i1<0 ) { std::cout<<"i1 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         #endif

         return this->p[ i0+i1*n_step[1] ];
     }


     //Read elements: for D=3
     inline T operator () (int i0, int i1, int i2) const 
     {
         #ifndef NDEBUG
         if(D != 3) { std::cout<<"Tensor_core::operator(int, int, int) only works for D=3 !!!"<<std::endl; exit(1); }
         if(i0 >= n[0] || i0<0 ) { std::cout<<"i0 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }   
         if(i1 >= n[1] || i1<0 ) { std::cout<<"i1 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         if(i2 >= n[2] || i2<0 ) { std::cout<<"i2 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         #endif

         return this->p[ i0+i1*n_step[1]+i2*n_step[2] ]; 
     }

     inline T& operator () (int i0, int i1, int i2)
     {
         #ifndef NDEBUG
         if(D != 3) { std::cout<<"Tensor_core::operator(int, int, int) only works for D=3 !!!"<<std::endl; exit(1); }
         if(i0 >= n[0] || i0<0 ) { std::cout<<"i0 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         if(i1 >= n[1] || i1<0 ) { std::cout<<"i1 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         if(i2 >= n[2] || i2<0 ) { std::cout<<"i2 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         #endif

         return this->p[ i0+i1*n_step[1]+i2*n_step[2] ];
     }


     //Read elements: for D>3
     template<typename... Values>
     T operator () (int i0, int i1, int i2, int i3, Values... inputs) const 
     {
         int vals[] = {inputs...};

         #ifndef NDEBUG
         int  len = sizeof...(Values);
         if(D != (len+4) ) { std::cout<<"Tensor_core::operator(int...) not consisten with D !!!"<<std::endl; exit(1); }
         if(i0 >= n[0] || i0<0 )   { std::cout<<"i0 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         if(i1 >= n[1] || i1<0 )   { std::cout<<"i1 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         if(i2 >= n[2] || i2<0 )   { std::cout<<"i2 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         if(i3 >= n[3] || i3<0 )   { std::cout<<"i3 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         for(int i=4; i<D; i++)
         {
             if( vals[i] >= n[i] || vals[i]<0  ) { std::cout<<"i... is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         }
         #endif

         int index = i0+i1*n_step[1]+i2*n_step[2]+i3*n_step[3];
         for(int i=4; i<D; i++) index += vals[i]*n_step[i];
         return this->p[index];
     }

     template<typename... Values>
     T& operator () (int i0, int i1, int i2, int i3, Values... inputs)
     {
         int vals[] = {inputs...};

         #ifndef NDEBUG
         int  len = sizeof...(Values);
         if(D != (len+4) ) { std::cout<<"Tensor_core::operator(int...) not consisten with D !!!"<<std::endl; exit(1); }
         if(i0 >= n[0] || i0<0 )   { std::cout<<"i0 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         if(i1 >= n[1] || i1<0 )   { std::cout<<"i1 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         if(i2 >= n[2] || i2<0 )   { std::cout<<"i2 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         if(i3 >= n[3] || i3<0 )   { std::cout<<"i3 is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         for(int i=4; i<D; i++)
         {
             if( vals[i] >= n[i] || vals[i]<0  ) { std::cout<<"i... is out of range in Tensor_core::operator() !!!"<<std::endl; exit(1); }
         }
         #endif

         int index = i0+i1*n_step[1]+i2*n_step[2]+i3*n_step[3];
         for(int i=4; i<D; i++) index += vals[i]*n_step[i];
         return this->p[index];
     }



  private:
     //Avoid program to generater constructor and assigment for Tensor_core. (Suppose to be an abstract class.)
     Tensor_core(const Tensor_core<T,D>& x)  { }
     Tensor_core<T,D> & operator  = (const Tensor_core<T,D>& x) { }

 };  //end class Tensor_core

} //end namespace tensor_hao

#endif
