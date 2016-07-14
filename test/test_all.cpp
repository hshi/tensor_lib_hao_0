#include<iostream>

#ifdef MPI_HAO
#include <mpi.h>
#endif

#ifdef USE_MAGMA
#include "magma.h"
#endif

using namespace std;

void Tensor_hao_ref_test();
void Tensor_hao_test();
void Tensor_core_test();
void Tensor_element_wise_test();
void Tensor_2d_common_fun_test();
void Tensor_2d_bl_cpu_test();

int main(int argc, char** argv)
{
    int rank=0;
#ifdef MPI_HAO
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

#ifdef USE_MAGMA
    magma_init();
#endif

    if(rank==0)  cout<<"\n\n\n=======Testing======="<<endl;
    Tensor_hao_ref_test();
    Tensor_hao_test();
    Tensor_core_test();
    Tensor_element_wise_test();
    Tensor_2d_common_fun_test();
    Tensor_2d_bl_cpu_test();

#ifdef USE_MAGMA
    magma_finalize();
#endif

#ifdef MPI_HAO
    MPI_Finalize();
#endif

    return 0;
}
