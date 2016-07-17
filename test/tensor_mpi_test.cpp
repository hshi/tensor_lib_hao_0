#ifdef MPI_HAO

#include "tensor_mpi.h"

using namespace std;
using namespace tensor_hao;

void MPIBcast_double_one_test()
{
    Tensor_hao<double,1> A(4), A_exact(4);

    A_exact = {14.861,12.630129,20.984,23.753129};

    int rank;MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank==0) A=A_exact;

    MPIBcast(A);

    size_t flag=diff(A,A_exact,1e-12);

    if(flag!=0) cout<<"Warning!!!!Bcast failed the double 1d test! rank: "<<rank<<endl;
}


void MPIBcast_complex_double_two_test()
{
    Tensor_hao<complex<double>,2> A(2,2), A_exact(2,2);

    A_exact = { {-13.769,40.877}, {-16.551971,38.73806},
                {-17.756,56.71},  {-22.838971, 66.77106} };

    int rank;MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank==0) A=A_exact;

    MPIBcast(A);

    size_t flag=diff(A,A_exact,1e-12);

    if(flag!=0) cout<<"Warning!!!!Bcast failed the complex double 2d test! rank: "<<rank<<endl;
}


void Tensor_mpi_test()
{
    int rank;MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0) cout<<"Testing Tensor_hao mpi version......\n"<<endl;

    MPIBcast_double_one_test();
    MPIBcast_complex_double_two_test();

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0) cout<<"PASSED! If these is no warning, we have passed all the test!"<<endl; 
    if(rank==0) cout<<endl; 
}


#endif
