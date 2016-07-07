#ifndef TENSOR_CORE_H
#define TENSOR_CORE_H

#include <cstdarg>
#include "tensor_base.h"

namespace tensor_hao
{

 template<class T = double, int D = 1> class Tensor_core : public Tensor_base<T>
 {
  protected:
     int n[D];
     int n_step[D];
  public:
     Tensor_core(void):Tensor_base<T>() {}
     ~Tensor_core() {}

     const int rank(int i) const {return n[i];}
     const int rank_step(int i) const {return n_step[i];}

  protected:
     Tensor_core<T, D> & operator  = (const std::initializer_list <T> &args)
     {
         int args_size = args.size();
         if(this->L != args_size) 
         {
             std::cout<<"Something is wrong with input args_size, not consisten with this->L! \n";
             std::cout<<this->L<<" "<<args_size<<std::endl;
             exit(1);
         }
         std::copy( args.begin(), args.begin()+args_size, this->p );
         return *this;
     }
 };

} //end namespace tensor_hao

#endif
