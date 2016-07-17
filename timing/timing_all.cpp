#include<iostream>

#ifdef MPI_HAO
#include <mpi.h>
#endif

#ifdef USE_MAGMA
#include "magma.h"
#endif

using namespace std;

#ifdef USE_MAGMA
void bl_cpu_magma_timing();
#endif

int main(int argc, char** argv)
{
    int rank=0;

#ifdef MPI_HAO
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if(rank==0)
    {
        cout<<"\n\nThis timing program compares time cost between CPU blas lapack and MAGMA blas lapack."<<endl;
        cout<<"The flag represents number of difference elements between results from CPU and these from MAGMA."<<endl;
        cout<<"It requires large memory and long computational time if -DUSE_MKL, else does nothing."<<endl;
        cout<<"Please submit the job if you are using a cluster, it takes ~10 minutes."<<endl;
        cout<<"\n=======Start timing======="<<endl;
        
    }

#ifdef USE_MAGMA
    magma_init();

    if(rank==0)
    {
        bl_cpu_magma_timing();
    }

    magma_finalize();
#endif

#ifdef MPI_HAO
    MPI_Finalize();
#endif

    return 0;
}
