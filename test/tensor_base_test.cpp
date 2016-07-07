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
        const int L=5;
        double p[L]={1.0,2.0,3.0,4.0,5.0};
        Tensor_base<double> base(L, p);

        cout<<base.size()<<endl;
        cout<<(base.data())[1]<<endl;
    }
}
