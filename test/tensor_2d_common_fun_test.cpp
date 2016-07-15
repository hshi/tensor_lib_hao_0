#ifdef MPI_HAO
#include <mpi.h>
#endif
#include <cmath>
#include "tensor_2d_common_fun.h"
#include "tensor_2d_bl_cpu.h"

using namespace std;
using namespace tensor_hao;

void Tensor_2d_trans_test()
{
    Tensor_hao<double, 2> A(2,3), B(3,2), B_exact(3,2);
    A={1.,2.,3.,4.,5.,6.};
    B_exact={1.,3.,5.,2.,4.,6.};
    B=trans(A);
    int flag = diff(B, B_exact, 1e-12);
    if(flag==0) cout<<"PASSED! Tensor 2d trans passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor 2d trans failed double test!"<<endl; 
}


void Tensor_2d_conjtrans_test()
{
    Tensor_hao<complex<double>, 2> A(2,3), B(3,2), B_exact(3,2);
    A={ {1.,1.}, {2.,2.}, {3.,3.}, {4.,4.}, {5.,5.}, {6.,6.} };
    B_exact={ {1.,-1.}, {3.,-3.}, {5.,-5.}, {2.,-2.}, {4.,-4.}, {6.,-6.} };
    B=conjtrans(A);
    int flag = diff(B, B_exact, 1e-12);
    if(flag==0) cout<<"PASSED! Tensor 2d conjtrans passed complex double test!"<<endl;
    else cout<<"WARNING!!!!Tensor 2d conjtrans failed complex double test!"<<endl;
}

void Tensor_2d_check_Hermitian_test()
{
    Tensor_hao<complex<double>,2> a(3,3);
    a = { {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
          {3.0,-4.0},   {2.0,0.0},    {5.123,3.11},
          {2.123,-3.11},{5.123,-3.11},{3,0.0}     };
    int flag=check_Hermitian(a);
    if(flag==0) cout<<"PASSED! Tensor 2d check_Hermitian passed test!"<<endl;
    else cout<<"WARNING!!!!Tensor 2d check_Hermitian failed test!"<<endl;
}

void LUDecomp_test()
{
    Tensor_hao<complex<double>,2> X(3,3);
    X = { {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
          {3.0,-2.0},   {2.0,0.0},    {5.123,3.11},
          {2.123,-5.11},{5.123,-6.11},{3,0.0} };

    LUDecomp<complex<double>> LU=LUconstruct_cpu(X);

    Tensor_hao<complex<double>,2> A_exact=LU.A;

    size_t flag=0;

    LUDecomp<complex<double>> LUC(LU);
    flag+=diff(LUC.A,A_exact,1e-13);

    LUDecomp<complex<double>> LUR(std::move(LU));
    flag+=diff(LUR.A,A_exact,1e-13);

    LUDecomp<complex<double>> LUEC;LUEC=LUC;
    flag+=diff(LUEC.A,A_exact,1e-13);

    LUDecomp<complex<double>> LUER;LUER=std::move(LUR);
    flag+=diff(LUER.A,A_exact,1e-13);


    if(flag==0) cout<<"PASSED! LUDecomp passed complex double test!"<<endl;
    else cout<<"WARNING!!!!!!!!! LUDecomp failed complex double test!"<<endl;
}

void determinant_test()
{
    Tensor_hao<complex<double>,2> X(3,3);
    X={ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11},
        {2.123,-5.11},{5.123,-6.11},{3,0.0}  };
    complex<double> det=determinant( LUconstruct_cpu(X) );
    complex<double> det_exact={123.11968700000003,3.3324580000000115};
    if(abs(det-det_exact)<1e-12) cout<<"PASSED! Determinant passed complex double test in cpu!"<<endl;
    else cout<<"WARNING!!!!!!!!! Determinant failed complex double test in cpu!"<<endl;
}


void lognorm_phase_determinant_test()
{
    Tensor_hao<complex<double>,2> X(3,3);
    X={ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11},
        {2.123,-5.11},{5.123,-6.11},{3,0.0}  };
    complex<double> lognorm, phase;
    lognorm_phase_determinant( LUconstruct_cpu(X), lognorm, phase );
    complex<double> lognorm_exact = 4.813523119460474;
    complex<double> phase_exact = {0.9996338948645466, 0.027056907397860167};
    int flag=0;
    if(abs(lognorm-lognorm_exact)>1e-12) flag++;
    if(abs(phase-phase_exact)>1e-12) flag++;
    if(flag==0) cout<<"PASSED! Lognorm_phase_determinant passed complex double test in cpu!"<<endl;
    else cout<<"WARNING!!!!!!!!! Lognorm_phase_determinant failed complex double test in cpu!"<<endl;
}

void log_determinant_test()
{
    Tensor_hao<complex<double>,2> X(3,3);


    X={ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11},
        {2.123,-5.11},{5.123,-6.11},{3,0.0}  };

    X*=1.e103;
    complex<double> logdet=log_determinant( LUconstruct_cpu(X) );
    complex<double> logdet_exact={716.3123168546207,0.027060209772387683};
    if(abs(logdet-logdet_exact)<1e-12) cout<<"PASSED! Log_determinant passed complex double test in cpu!"<<endl;
    else cout<<"WARNING!!!!!!!!! Log_determinant failed complex double test in cpu!"<<endl;
}

void D_Multi_Matrix_test()
{
    Tensor_hao<complex<double>,2> A(3,2);
    Tensor_hao<complex<double>,1> D(3);

    A = { {2.0,0.0} ,   {3.0,5.0},    {3.123,3.11},
          {3.0,-6.0},   {2.0,1.0},    {6.123,3.11} };

    D = { {1.2,0.0}, {2.0,0.0}, {3.0,0.0} };

    Tensor_hao<complex<double>,2> B=D_Multi_Matrix(D,A);

    Tensor_hao<complex<double>,2> B_exact(3,2);
    B_exact = { {2.4,0.0} ,   {6.0,10.0},    {9.369,9.33},
                {3.6,-7.2},   {4.0,2.0 },    {18.369,9.33} };

    int flag=diff(B,B_exact,1e-12);
    if(flag==0) cout<<"PASSED! D_Multi_Matrix passed complex double test!"<<endl;
    else cout<<"WARNING!!!!!!!!! D_Multi_Matrix failed complex double test!"<<endl;
}

void Tensor_2d_check_skew_symmetric_test()
{
    Tensor_hao<complex<double>,2> a(3,3);
    a = { {0.0,0.0} ,    {3.0,4.0},     {2.123,3.11},
          {-3.0,-4.0},   {0.0,0.0},     {5.123,3.11},
          {-2.123,-3.11},{-5.123,-3.11},{0,0.0}     };
    int flag=check_skew_symmetric(a);
    if(flag==0) cout<<"PASSED! Tensor 2d check_skew_symmetric passed test!"<<endl;
    else cout<<"WARNING!!!!Tensor 2d check_skew_symmetric failed test!"<<endl;
}

void pfaffian_test()
{
    Tensor_hao<complex<double>,2> A(4,4);
    A = { {0.0,0.0},  {1.0,2.0},  {1.5,0.0}, {2.3,0.0},
          {-1.0,-2.0},{0.0,0.0},  {-3.0,0.0},{1.5,0.0},
          {-1.5,0.0}, {3.0,0.0},  {0.0,0.0}, {-2.5,-5.3},
          {-2.3,0.0}, {-1.5,0.0}, {2.5,5.3}, {0.0,0.0} };
    complex<double> pf=Pfaffian(A);
    complex<double> exact{-1.05,-10.3};
    if( abs(pf-exact)< 1e-13 ) cout<<"PASSED! Pfaffian passed complex double test!"<<endl;
    else cout<<"WARNING!!!!!!!!! Pfaffian failed complex double test!"<<endl;
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
        Tensor_2d_check_Hermitian_test();
        LUDecomp_test();
        determinant_test();
        lognorm_phase_determinant_test();
        log_determinant_test();
        D_Multi_Matrix_test();
        Tensor_2d_check_skew_symmetric_test();
        pfaffian_test();
    }

}

