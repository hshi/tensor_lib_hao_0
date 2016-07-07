#ifndef TENSOR_BASE_H
#define TENSOR_BASE_H

#include <iostream>
#include <vector>
namespace tensor_hao
{

 //Tensor_base class, it is suppose to be abstract class.
 template<class T> class Tensor_base
 {
  protected:
     int L;
     T* p;
  public:
     Tensor_base(void): L(0), p(nullptr) {}
     Tensor_base(int L_in, T* p_in): L(L_in), p(p_in) {}
     ~Tensor_base() {}

     //=========
     //FUNCTIONS
     //=========
 
     const int size() const {return L;}

     const T * data() const {return p;}
           T * data()       {return p;}

     //Return reference to pointer
     const T *& data_ref() const {return p;}
           T *& data_ref()       {return p;}

  protected:
     void copy_list(const std::initializer_list <T> &args)
     {
         int args_size = args.size();
         if(L != args_size)
         {
             std::cout<<"Something is wrong with input args_size, not consisten with L! \n";
             std::cout<<L<<" "<<args_size<<std::endl;
             exit(1);
         }
         std::copy( args.begin(), args.begin()+args_size, p );
     }

 }; //end class Tensor_base


} //end namespace tensor_hao

#endif
