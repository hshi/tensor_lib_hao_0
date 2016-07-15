#ifdef MPI_HAO
#include <mpi.h>
#endif
#include <cmath>
#include "tensor_hao_ref.h"
#include "tensor_hao.h"

using namespace std;
using namespace tensor_hao;

void Tensor_hao_void_constructor_test()
{
    Tensor_hao<double,2>  tensor;
    int flag=0;
    if(tensor.data() ) flag++;
    if(tensor.size() !=0 ) flag++;

    if(flag==0) cout<<"PASSED! Tensor_hao void constructor passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_hao void constructor failed double test!"<<endl;
}

void Tensor_hao_variadic_constructor_test()
{
    const int D=3;
    Tensor_hao<double, D>  tensor(3,4,7);
    int size=84;
    int n[D]={3,4,7};
    int n_step[D]={1,3,12};

    int flag=0;

    if( !tensor.data() ) flag++;
    if(tensor.size() != size ) flag++;
    for(int i=0; i<D; i++)  {if( tensor.rank(i) != n[i] ) flag++;}
    for(int i=0; i<D; i++)  {if( tensor.rank_step(i) != n_step[i] ) flag++;}

    if(flag==0) cout<<"PASSED! Tensor_hao variadic constructor passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_hao variadic constructor failed double test!"<<endl;
}


void Tensor_hao_pointer_constructor_test()
{
    const int D=3;
    int n_ptr[D] = {3,4,7};
    Tensor_hao<double, D>  tensor(n_ptr);
    int size=84;
    int n[D]={3,4,7};
    int n_step[D]={1,3,12};

    int flag=0;

    if( !tensor.data() ) flag++;
    if(tensor.size() != size ) flag++;
    for(int i=0; i<D; i++)  {if( tensor.rank(i) != n[i] ) flag++;}
    for(int i=0; i<D; i++)  {if( tensor.rank_step(i) != n_step[i] ) flag++;}

    if(flag==0) cout<<"PASSED! Tensor_hao pointer constructor passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_hao pointer constructor failed double test!"<<endl;
}

void Tensor_hao_copy_constructor_test()
{
    Tensor_hao<double,2>  tensor_a(3,4);
    tensor_a={1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,7.0 ,8.0 ,9.0 ,10.0 ,11.0 ,12.0 };

    Tensor_hao<double,2>  tensor_b( tensor_a );

    const int size=12;
    const int D=2;
    int n[D]={3,4};
    int n_step[D]={1,3};
    double p_value[size]={1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,7.0 ,8.0 ,9.0 ,10.0 ,11.0 ,12.0 };

    int flag=0;
    double* p=tensor_b.data();
    if( tensor_b.size() != size ) flag++;
    for(int i=0; i<D; i++)  {if( tensor_b.rank(i) != n[i] ) flag++;}
    for(int i=0; i<D; i++)  {if( tensor_b.rank_step(i) != n_step[i] ) flag++;}
    for(int i=0; i<size; i++)  { if(  std::abs(p[i] - p_value[i]) > 1e-12 ) flag++; }
    if(flag==0) cout<<"PASSED! Tensor_hao copy constructor passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_hao copy constructor failed double test!"<<endl;

}

void Tensor_hao_move_constructor_test()
{
    Tensor_hao<double,2>  tensor_a(3,4);
    tensor_a={1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,7.0 ,8.0 ,9.0 ,10.0 ,11.0 ,12.0 };

    Tensor_hao<double,2>  tensor_b( move(tensor_a) );

    const int size=12;
    const int D=2;
    int n[D]={3,4};
    int n_step[D]={1,3};
    double p_value[size]={1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,7.0 ,8.0 ,9.0 ,10.0 ,11.0 ,12.0 };

    int flag=0;
    double* p=tensor_b.data();
    if( tensor_b.size() != size ) flag++;
    for(int i=0; i<D; i++)  {if( tensor_b.rank(i) != n[i] ) flag++;}
    for(int i=0; i<D; i++)  {if( tensor_b.rank_step(i) != n_step[i] ) flag++;}
    for(int i=0; i<size; i++)  { if(  std::abs(p[i] - p_value[i]) > 1e-12 ) flag++; }
    if(flag==0) cout<<"PASSED! Tensor_hao move constructor passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_hao move constructor failed double test!"<<endl;

}

void Tensor_hao_copy_ref_constructor_test()
{
    Tensor_hao<double,2>  tensor_a(3,4);
    tensor_a={1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,7.0 ,8.0 ,9.0 ,10.0 ,11.0 ,12.0 };
    Tensor_hao_ref<double,2>  tensor_a_ref(tensor_a);

    Tensor_hao<double,2>  tensor_b( tensor_a_ref );

    const int size=12;
    const int D=2;
    int n[D]={3,4};
    int n_step[D]={1,3};
    double p_value[size]={1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,7.0 ,8.0 ,9.0 ,10.0 ,11.0 ,12.0 };

    int flag=0;
    double* p=tensor_b.data();
    if( tensor_b.size() != size ) flag++;
    for(int i=0; i<D; i++)  {if( tensor_b.rank(i) != n[i] ) flag++;}
    for(int i=0; i<D; i++)  {if( tensor_b.rank_step(i) != n_step[i] ) flag++;}
    for(int i=0; i<size; i++)  { if(  std::abs(p[i] - p_value[i]) > 1e-12 ) flag++; }
    if(flag==0) cout<<"PASSED! Tensor_hao copy ref constructor passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_hao copy ref constructor failed double test!"<<endl;

}

void Tensor_hao_copy_assignment_test()
{
    Tensor_hao<double,2>  tensor_a(3,4);
    tensor_a={1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,7.0 ,8.0 ,9.0 ,10.0 ,11.0 ,12.0 };

    Tensor_hao<double,2>  tensor_b; tensor_b = tensor_a;

    const int size=12;
    const int D=2;
    int n[D]={3,4};
    int n_step[D]={1,3};
    double p_value[size]={1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,7.0 ,8.0 ,9.0 ,10.0 ,11.0 ,12.0 };

    int flag=0;
    double* p=tensor_b.data();
    if( tensor_b.size() != size ) flag++;
    for(int i=0; i<D; i++)  {if( tensor_b.rank(i) != n[i] ) flag++;}
    for(int i=0; i<D; i++)  {if( tensor_b.rank_step(i) != n_step[i] ) flag++;}
    for(int i=0; i<size; i++)  { if(  std::abs(p[i] - p_value[i]) > 1e-12 ) flag++; }
    if(flag==0) cout<<"PASSED! Tensor_hao copy assignment passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_hao copy assignment failed double test!"<<endl;

}

void Tensor_hao_move_assignment_test()
{
    Tensor_hao<double,2>  tensor_a(3,4);
    tensor_a={1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,7.0 ,8.0 ,9.0 ,10.0 ,11.0 ,12.0 };

    Tensor_hao<double,2>  tensor_b; tensor_b = move(tensor_a) ;

    const int size=12;
    const int D=2;
    int n[D]={3,4};
    int n_step[D]={1,3};
    double p_value[size]={1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,7.0 ,8.0 ,9.0 ,10.0 ,11.0 ,12.0 };

    int flag=0;
    double* p=tensor_b.data();
    if( tensor_b.size() != size ) flag++;
    for(int i=0; i<D; i++)  {if( tensor_b.rank(i) != n[i] ) flag++;}
    for(int i=0; i<D; i++)  {if( tensor_b.rank_step(i) != n_step[i] ) flag++;}
    for(int i=0; i<size; i++)  { if(  std::abs(p[i] - p_value[i]) > 1e-12 ) flag++; }
    if(flag==0) cout<<"PASSED! Tensor_hao move assignment passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_hao move assignment failed double test!"<<endl;

}

void Tensor_hao_copy_ref_assignment_test()
{
    Tensor_hao<double,2>  tensor_a(3,4);
    tensor_a={1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,7.0 ,8.0 ,9.0 ,10.0 ,11.0 ,12.0 };
    Tensor_hao_ref<double,2>  tensor_a_ref(tensor_a);

    Tensor_hao<double,2>  tensor_b; tensor_b = tensor_a_ref ;

    const int size=12;
    const int D=2;
    int n[D]={3,4};
    int n_step[D]={1,3};
    double p_value[size]={1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,7.0 ,8.0 ,9.0 ,10.0 ,11.0 ,12.0 };

    int flag=0;
    double* p=tensor_b.data();
    if( tensor_b.size() != size ) flag++;
    for(int i=0; i<D; i++)  {if( tensor_b.rank(i) != n[i] ) flag++;}
    for(int i=0; i<D; i++)  {if( tensor_b.rank_step(i) != n_step[i] ) flag++;}
    for(int i=0; i<size; i++)  { if(  std::abs(p[i] - p_value[i]) > 1e-12 ) flag++; }
    if(flag==0) cout<<"PASSED! Tensor_hao copy ref assignment passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_hao copy ref assignment failed double test!"<<endl;

}

void Tensor_hao_slice_test()
{
    Tensor_hao<double,3>  tensor(3,4,5);
    int L = tensor.size(); double* p = tensor.data();
    for(int i=0; i<L; i++) p[i] = i*1.0;

    Tensor_hao_ref<double,2 >  slice = tensor[4];

    int flag=0;
    int slice_L = slice.size(); double* slice_p = slice.data();
    for(int i=0; i<slice_L; i++)
    {
        if( std::abs( slice_p[i]- (i+12*4.0) ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor_hao slice function passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_hao slice function failed double test!"<<endl;

}


void Tensor_hao_test()
{
    int rank=0;
#ifdef MPI_HAO
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if(rank==0)
    {
        Tensor_hao_void_constructor_test();
        Tensor_hao_variadic_constructor_test();
        Tensor_hao_pointer_constructor_test();
        Tensor_hao_copy_constructor_test();
        Tensor_hao_move_constructor_test();
        Tensor_hao_copy_ref_constructor_test();
        Tensor_hao_copy_assignment_test();
        Tensor_hao_move_assignment_test();
        Tensor_hao_copy_ref_assignment_test();
        Tensor_hao_slice_test();
    }

}
