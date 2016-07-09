#ifdef MPI_HAO
#include <mpi.h>
#endif
#include "tensor_base.h"

using namespace std;
using namespace tensor_hao;

void Tensor_base_test()
{
    int rank=0;
#ifdef MPI_HAO
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if(rank==0)
    {
    }
}
