#ifdef MPI_HAO
#include <mpi.h>
#endif
#include <cmath>
#include "tensor_2d_common_fun.h"

using namespace std;
using namespace tensor_hao;

void Tensor_2d_trans_test()
{
    Tensor_hao<double, 2> A(2,3), B(3,2), B_exact(3,2);
    A={1.,2.,3.,4.,5.,6.};
    B_exact={1.,3.,5.,2.,4.,6.};
    B=trans(A);
    int flag = diff(B, B_exact, 1e-12);
    if(flag==0) cout<<"Tensor 2d trans passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor 2d trans failed double test!"<<endl; 
}


void Tensor_2d_conjtrans_test()
{
    Tensor_hao<complex<double>, 2> A(2,3), B(3,2), B_exact(3,2);
    A={ {1.,1.}, {2.,2.}, {3.,3.}, {4.,4.}, {5.,5.}, {6.,6.} };
    B_exact={ {1.,-1.}, {3.,-3.}, {5.,-5.}, {2.,-2.}, {4.,-4.}, {6.,-6.} };
    B=conjtrans(A);
    int flag = diff(B, B_exact, 1e-12);
    if(flag==0) cout<<"Tensor 2d conjtrans passed complex double test!"<<endl;
    else cout<<"WARNING!!!!Tensor 2d conjtrans failed complex double test!"<<endl;
}



void Tensor_2d_common_fun_test()
{
    int rank=0;
#ifdef MPI_HAO
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if(rank==0)
    {
        Tensor_2d_trans_test();
        Tensor_2d_conjtrans_test();
    }

}

