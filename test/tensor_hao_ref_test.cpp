#ifdef MPI_HAO
#include <mpi.h>
#endif
#include <cmath>
#include "tensor_hao.h"
#include "tensor_hao_ref.h"

using namespace std;
using namespace tensor_hao;

void Tensor_hao_ref_void_constructor_test()
{
    Tensor_hao_ref<double,2>  tensor_ref;
    int flag=0;
    if(tensor_ref.data() ) flag++;
    if(tensor_ref.size() !=0 ) flag++;

    if(flag==0) cout<<"PASSED! Tensor_hao_ref void constructor passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_hao_ref void constructor failed double test!"<<endl; 
}

void Tensor_hao_ref_variadic_constructor_test()
{
    const int D=3;
    Tensor_hao_ref<double, D>  tensor_ref(3,4,7);
    int size=84;
    int n[D]={3,4,7};
    int n_step[D]={1,3,12};

    int flag=0;

    if(tensor_ref.data() ) flag++;
    if(tensor_ref.size() != size ) flag++;
    for(int i=0; i<D; i++)  {if( tensor_ref.rank(i) != n[i] ) flag++;} 
    for(int i=0; i<D; i++)  {if( tensor_ref.rank_step(i) != n_step[i] ) flag++;} 

    if(flag==0) cout<<"PASSED! Tensor_hao_ref variadic constructor passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_hao_ref variadic constructor failed double test!"<<endl;
}

void Tensor_hao_ref_pointer_constructor_test()
{
    const int D=3;
    int n_ptr[D] = {3,4,7};
    Tensor_hao_ref<double, D>  tensor_ref(n_ptr);
    int size=84;
    int n[D]={3,4,7};
    int n_step[D]={1,3,12};

    int flag=0;

    if(tensor_ref.data() ) flag++;
    if(tensor_ref.size() != size ) flag++;
    for(int i=0; i<D; i++)  {if( tensor_ref.rank(i) != n[i] ) flag++;}
    for(int i=0; i<D; i++)  {if( tensor_ref.rank_step(i) != n_step[i] ) flag++;}

    if(flag==0) cout<<"PASSED! Tensor_hao_ref pointer constructor passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_hao_ref pointer constructor failed double test!"<<endl;
}


void Tensor_hao_ref_constructor_assginment_test()
{
    Tensor_hao<double,2>  tensor_a(3,4); 

    Tensor_hao_ref<double,2> tensor_a_ref_p0(tensor_a);
    tensor_a_ref_p0={1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,7.0 ,8.0 ,9.0 ,10.0 ,11.0 ,12.0 };

    Tensor_hao_ref<double,2> tensor_a_ref_p1(tensor_a_ref_p0);

    Tensor_hao_ref<double,2> tensor_a_ref_p2, tensor_a_ref_p3;

    tensor_a_ref_p2 = tensor_a_ref_p0; 

    tensor_a_ref_p3 = tensor_a; 

    const int size=12; 
    const int D=2;
    int n[D]={3,4};
    int n_step[D]={1,3};
    double p_value[size]={1.0 ,2.0 ,3.0 ,4.0 ,5.0 ,6.0 ,7.0 ,8.0 ,9.0 ,10.0 ,11.0 ,12.0 };

    int flag=0;

    if( tensor_a_ref_p0.data() != tensor_a.data() ) flag++;
    if( tensor_a_ref_p1.data() != tensor_a.data() ) flag++;
    if( tensor_a_ref_p2.data() != tensor_a.data() ) flag++;
    if( tensor_a_ref_p3.data() != tensor_a.data() ) flag++;

    double* p=tensor_a_ref_p0.data();
    if( tensor_a_ref_p3.size() != size ) flag++;
    for(int i=0; i<D; i++)  {if( tensor_a_ref_p3.rank(i) != n[i] ) flag++;}
    for(int i=0; i<D; i++)  {if( tensor_a_ref_p3.rank_step(i) != n_step[i] ) flag++;}
    for(int i=0; i<size; i++)  { if(  std::abs(p[i] - p_value[i]) > 1e-12 ) flag++; }
    if(flag==0) cout<<"PASSED! Tensor_hao_ref constructor and assignment passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_hao_ref constructor and assignment failed double test!"<<endl;

}

void Tensor_hao_ref_point_test()
{
    Tensor_hao_ref<double,2> tensor_ref(5,8), tensor_ref_p(5,8);
    vector<double> vec(40);
    tensor_ref.point(vec);
    tensor_ref_p.point( vec.data() );

    int flag=0;
    if(tensor_ref.data()   != vec.data() ) flag++;
    if(tensor_ref_p.data() != vec.data() ) flag++;

    if(flag==0) cout<<"PASSED! Tensor_hao_ref point function passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_hao_ref point function failed double test!"<<endl;
}

void Tensor_hao_ref_slice_test()
{
    Tensor_hao_ref<double,3>  tensor(3,4,5);
    int L = tensor.size(); vector<double> p(L);
    for(int i=0; i<L; i++) p[i] = i*1.0;
    tensor.point(p);

    Tensor_hao_ref<double,2 >  slice = tensor[4];

    int flag=0;
    int slice_L = slice.size(); double* slice_p = slice.data();
    for(int i=0; i<slice_L; i++) 
    {
        if( std::abs( slice_p[i]- (i+12*4.0) ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor_hao_ref slice function passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_hao_ref slice function failed double test!"<<endl;

}

void Tensor_hao_ref_test()
{
    int rank=0;
#ifdef MPI_HAO
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if(rank==0)
    {
        Tensor_hao_ref_void_constructor_test();
        Tensor_hao_ref_variadic_constructor_test();
        Tensor_hao_ref_pointer_constructor_test();
        Tensor_hao_ref_constructor_assginment_test();
        Tensor_hao_ref_point_test();
        Tensor_hao_ref_slice_test();
        cout<<endl;
    }
}
