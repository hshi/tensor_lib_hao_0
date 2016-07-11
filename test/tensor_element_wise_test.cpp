#ifdef MPI_HAO
#include <mpi.h>
#endif
#include <cmath>
#include "tensor_element_wise.h"

using namespace std;
using namespace tensor_hao;

void tensor_conj_test()
{
    Tensor_hao< complex<double>,2>  tensor_a(3,2);
    tensor_a={ {1.0 ,2.0} , {3.0 ,4.0} , {5.0 ,6.0} , {7.0 ,8.0} , {9.0 ,10.0} , {11.0 ,12.0} };

    Tensor_hao< complex<double>,2> tensor_b = conj( tensor_a );

    int flag=0; 
    for(int j=0; j<2; j++)
    {
        for(int i=0; i<3; i++)
        {
            if( std::abs( tensor_b(i,j)-conj( tensor_a(i,j) ) ) >1e-12 ) flag++;
        }
    }
    if(flag==0) cout<<"Tensor conj passed complex<double> test!"<<endl;
    else cout<<"WARNING!!!!Tensor conj failed complex<double> test!"<<endl;

}

void tensor_exp_test()
{
    Tensor_hao<double,2>  tensor_a(3,4);
    tensor_a={1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

    Tensor_hao<double,2> tensor_b = exp( tensor_a );

    int flag=0;
    for(int j=0; j<4; j++)
    {
        for(int i=0; i<3; i++)
        {
            if( std::abs( tensor_b(i,j)-exp( tensor_a(i,j) ) ) >1e-12 ) flag++;
        }
    }
    if(flag==0) cout<<"Tensor exp passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor exp failed double test!"<<endl;

}

void tensor_minus_test()
{
    Tensor_hao<double,2>  tensor_a(3,4);
    tensor_a={1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

    Tensor_hao<double,2> tensor_b = -tensor_a;

    int flag=0;
    for(int j=0; j<4; j++)
    {
        for(int i=0; i<3; i++)
        {
            if( std::abs( tensor_b(i,j) + tensor_a(i,j) ) >1e-12 ) flag++;
        }
    }
    if(flag==0) cout<<"Tensor minus passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor minus failed double test!"<<endl;

}

void Tensor_element_wise_test()
{
    int rank=0;
#ifdef MPI_HAO
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if(rank==0)
    {
        tensor_conj_test();
        tensor_exp_test();
        tensor_minus_test();
    }
}
