#ifdef USE_MAGMA

#ifdef MPI_HAO
#include <mpi.h>
#endif
#include <cmath>
#include "tensor_2d_bl_magma.h"

using namespace std;
using namespace tensor_hao;

void gmm_magma_float_test()
{
    Tensor_hao<float,2> a(2,3), b(3,2), c(2,2), c_exact(2,2);
    a={0.0,3.0,2.123,2.0,4.0,3.123 };
    b={0.0,3.0,2.123,2.0,4.0,3.123 };
    c_exact={14.861,12.630129,20.984,23.753129};
    gmm_magma(a,b,c);
    //gmm_magma(trans(a),b,c, 'T');
    //gmm_magma(trans(a), trans(b),c, 'T', 'T');
    //gmm_magma(trans(a), trans(b),c, 'C', 'T');
    //gmm_magma(trans(a), trans(b),c, 'C', 'C');

    int flag=diff(c,c_exact,1e-5);
    if(flag==0) cout<<"PASSED! Gmm_magma passed float test!"<<endl;
    else cout<<"WARNING!!!!!!!!! Gmm_magma failed float test!"<<endl;
}

void gmm_magma_double_test()
{
    Tensor_hao<double,2> a(2,3), b(3,2), c(2,2), c_exact(2,2);
    a={0.0,3.0,2.123,2.0,4.0,3.123 };
    b={0.0,3.0,2.123,2.0,4.0,3.123 };
    c_exact={14.861,12.630129,20.984,23.753129};
    gmm_magma(a,b,c);
    //gmm_magma(trans(a),b,c, 'T');
    //gmm_magma(trans(a), trans(b),c, 'T', 'T');
    //gmm_magma(trans(a), trans(b),c, 'C', 'T');
    //gmm_magma(trans(a), trans(b),c, 'C', 'C');

    int flag=diff(c,c_exact,1e-12);
    if(flag==0) cout<<"PASSED! Gmm_magma passed double test!"<<endl;
    else cout<<"WARNING!!!!!!!!! Gmm_magma failed double test!"<<endl;
}


void gmm_magma_complex_float_test()
{
    Tensor_hao<complex<float>,2> a(2,3), b(3,2), c(2,2), c_exact(2,2);
    a={ {0.0,0.8},{3.0,4.0},{2.123,3.11},{2.0,3.3},{4.0,5.0},{3.123,4.11} };
    b={ {0.0,0.8},{3.0,4.0},{2.123,3.11},{2.0,3.3},{4.0,5.0},{3.123,4.11} };
    c_exact={ {-13.769,40.877}, {-16.551971,38.73806}, {-17.756,56.71}, {-22.838971, 66.77106} };
    gmm_magma(a,b,c);
    //gmm_magma(trans(a),b,c, 'T');
    //gmm_magma(trans(a), trans(b),c, 'T', 'T');
    //gmm_magma(conjtrans(a), trans(b),c, 'C', 'T');
    //gmm_magma(conjtrans(a), conjtrans(b),c, 'C', 'C');

    int flag=diff(c,c_exact,1e-5);
    if(flag==0) cout<<"PASSED! Gmm_magma passed complex float test!"<<endl;
    else cout<<"WARNING!!!!!!!!! Gmm_magma failed complex float test!"<<endl;
}

void gmm_magma_complex_double_test()
{
    Tensor_hao<complex<double>,2> a(2,3), b(3,2), c(2,2), c_exact(2,2);
    a={ {0.0,0.8},{3.0,4.0},{2.123,3.11},{2.0,3.3},{4.0,5.0},{3.123,4.11} };
    b={ {0.0,0.8},{3.0,4.0},{2.123,3.11},{2.0,3.3},{4.0,5.0},{3.123,4.11} };
    c_exact={ {-13.769,40.877}, {-16.551971,38.73806}, {-17.756,56.71}, {-22.838971, 66.77106} };
    gmm_magma(a,b,c);
    //gmm_magma(trans(a),b,c, 'T');
    //gmm_magma(trans(a), trans(b),c, 'T', 'T');
    //gmm_magma(conjtrans(a), trans(b),c, 'C', 'T');
    //gmm_magma(conjtrans(a), conjtrans(b),c, 'C', 'C');

    int flag=diff(c,c_exact,1e-12);
    if(flag==0) cout<<"PASSED! Gmm_magma passed complex double test!"<<endl;
    else cout<<"WARNING!!!!!!!!! Gmm_magma failed complex double test!"<<endl;
}

void Tensor_2d_bl_magma_test()
{
    int rank=0;
#ifdef MPI_HAO
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if(rank==0)
    {
        gmm_magma_float_test();
        gmm_magma_double_test();
        gmm_magma_complex_float_test();
        gmm_magma_complex_double_test();
        //eigen_magma_double_test();
        //eigen_magma_complex_double_test();
        //LUconstruct_magma_test();
        //inverse_magma_test();
        //solve_lineq_magma_test();
        //QRMatrix_magma_test();
        //SVDMatrix_magma_test();
        cout<<endl;
    }
}

#endif
