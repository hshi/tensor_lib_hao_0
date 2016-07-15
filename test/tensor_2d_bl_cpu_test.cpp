#ifdef MPI_HAO
#include <mpi.h>
#endif
#include <cmath>
#include "tensor_2d_bl_cpu.h"

using namespace std;
using namespace tensor_hao;

void gmm_cpu_float_test()
{
    Tensor_hao<float,2> a(2,3), b(3,2), c(2,2), c_exact(2,2);
    a={0.0,3.0,2.123,2.0,4.0,3.123 };
    b={0.0,3.0,2.123,2.0,4.0,3.123 };
    c_exact={14.861,12.630129,20.984,23.753129};
    gmm_cpu(a,b,c);
    //gmm_cpu(trans(a),b,c, 'T');
    //gmm_cpu(trans(a), trans(b),c, 'T', 'T');
    //gmm_cpu(trans(a), trans(b),c, 'C', 'T');
    //gmm_cpu(trans(a), trans(b),c, 'C', 'C');

    int flag=diff(c,c_exact,1e-5);
    if(flag==0) cout<<"PASSED! Gmm_cpu passed float test!"<<endl;
    else cout<<"WARNING!!!!!!!!! Gmm_cpu failed float test!"<<endl;
}

void gmm_cpu_double_test()
{
    Tensor_hao<double,2> a(2,3), b(3,2), c(2,2), c_exact(2,2);
    a={0.0,3.0,2.123,2.0,4.0,3.123 };
    b={0.0,3.0,2.123,2.0,4.0,3.123 };
    c_exact={14.861,12.630129,20.984,23.753129};
    gmm_cpu(a,b,c);
    //gmm_cpu(trans(a),b,c, 'T');
    //gmm_cpu(trans(a), trans(b),c, 'T', 'T');
    //gmm_cpu(trans(a), trans(b),c, 'C', 'T');
    //gmm_cpu(trans(a), trans(b),c, 'C', 'C');

    int flag=diff(c,c_exact,1e-12);
    if(flag==0) cout<<"PASSED! Gmm_cpu passed double test!"<<endl;
    else cout<<"WARNING!!!!!!!!! Gmm_cpu failed double test!"<<endl;
}


void gmm_cpu_complex_float_test()
{
    Tensor_hao<complex<float>,2> a(2,3), b(3,2), c(2,2), c_exact(2,2);
    a={ {0.0,0.8},{3.0,4.0},{2.123,3.11},{2.0,3.3},{4.0,5.0},{3.123,4.11} };
    b={ {0.0,0.8},{3.0,4.0},{2.123,3.11},{2.0,3.3},{4.0,5.0},{3.123,4.11} };
    c_exact={ {-13.769,40.877}, {-16.551971,38.73806}, {-17.756,56.71}, {-22.838971, 66.77106} };
    gmm_cpu(a,b,c);
    //gmm_cpu(trans(a),b,c, 'T');
    //gmm_cpu(trans(a), trans(b),c, 'T', 'T');
    //gmm_cpu(conjtrans(a), trans(b),c, 'C', 'T');
    //gmm_cpu(conjtrans(a), conjtrans(b),c, 'C', 'C');

    int flag=diff(c,c_exact,1e-5);
    if(flag==0) cout<<"PASSED! Gmm_cpu passed complex float test!"<<endl;
    else cout<<"WARNING!!!!!!!!! Gmm_cpu failed complex float test!"<<endl;
}

void gmm_cpu_complex_double_test()
{
    Tensor_hao<complex<double>,2> a(2,3), b(3,2), c(2,2), c_exact(2,2);
    a={ {0.0,0.8},{3.0,4.0},{2.123,3.11},{2.0,3.3},{4.0,5.0},{3.123,4.11} };
    b={ {0.0,0.8},{3.0,4.0},{2.123,3.11},{2.0,3.3},{4.0,5.0},{3.123,4.11} };
    c_exact={ {-13.769,40.877}, {-16.551971,38.73806}, {-17.756,56.71}, {-22.838971, 66.77106} };
    gmm_cpu(a,b,c);
    //gmm_cpu(trans(a),b,c, 'T');
    //gmm_cpu(trans(a), trans(b),c, 'T', 'T');
    //gmm_cpu(conjtrans(a), trans(b),c, 'C', 'T');
    //gmm_cpu(conjtrans(a), conjtrans(b),c, 'C', 'C');

    int flag=diff(c,c_exact,1e-12);
    if(flag==0) cout<<"PASSED! Gmm_cpu passed complex double test!"<<endl;
    else cout<<"WARNING!!!!!!!!! Gmm_cpu failed complex double test!"<<endl;
}

void eigen_cpu_double_test()
{
    Tensor_hao<double,2> a(3,3), a_exact(3,3);
    Tensor_hao<double,1> w(3), w_exact(3);

    a = { 1.0, 3.0, 2.123,  
          3.0, 2.0, 5.123,  
          2.123, 5.123, 3 };

    a_exact = { 0.2849206371113407, -0.7715217080565622,    0.5688360788008725,
                0.8701970661567061, -0.040665575250991126, -0.4910227866828251,
               -0.40196678544418185,-0.6349020121144606,   -0.6597894652180195 };

    w_exact = {-2.8850331801092803, -0.33813149591901365, 9.223164676028293 };

    eigen_cpu(a,w);

    int flag=0;

    double* p_a = a.data();
    double* p_a_exact = a_exact.data();
    for(int i=0; i<a.size(); i++)
    {
        if( abs( abs(p_a[i]) - abs(p_a_exact[i]) )>1e-13 ) flag++;
    }
    flag+=diff(w,w_exact,1e-13);
    if(flag==0) cout<<"PASSED! Eigen_cpu passed double symmetric test!"<<endl;
    else cout<<"WARNING!!!!!!!!! Eigen_cpu failed double symmetric test!"<<endl;
}


void eigen_cpu_complex_double_test()
{
    Tensor_hao<complex<double>,2> a(3,3), a_exact(3,3);
    Tensor_hao<double,1> w(3), w_exact(3);

    a = { {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
          {3.0,-4.0},   {2.0,0.0},    {5.123,3.11},
          {2.123,-3.11},{5.123,-3.11},{3,0.0}     };

    a_exact = { {-0.4053433965286621, -0.3217472918461721},
                {-0.3733963692733272,  0.6060804552476304},
                {0.47478104875888194,  0},
                {0.13035873463974057,  0.6902772720595061},
                {-0.26751344366934643,-0.20279279787239068},
                {0.6275631654012745,   0},
                {-0.179307184764388,   0.4544757777410628},
                {-0.5593786354476359,  0.26009385608337265},
                {-0.6170473475925071,  0} };

    w_exact = { -4.7040348985237666,-1.1586196209127053,11.862654519436473 }; 

    eigen_cpu(a,w);

    int flag=0;

    complex<double>* p_a = a.data();
    complex<double>* p_a_exact = a_exact.data();
    for(int i=0; i<a.size(); i++)
    {
        if( abs( abs(p_a[i]) - abs(p_a_exact[i]) )>1e-13 ) flag++;
    }
    flag+=diff(w,w_exact,1e-13);
    if(flag==0) cout<<"PASSED! Eigen_cpu passed complex double hermition test!"<<endl;
    else cout<<"WARNING!!!!!!!!! Eigen_cpu failed complex double hermition test!"<<endl;
}

void LUconstruct_cpu_test()
{
    Tensor_hao<complex<double>,2> X(3,3), A_exact(3,3);

    X={ {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
        {3.0,-2.0},   {2.0,0.0},    {5.123,3.11},
        {2.123,-5.11},{5.123,-6.11},{3,0.0}   };

    A_exact={ {3,4} ,       {0.75236,0.03351999999999994},  {0.12,-0.16},
              {2,0},        {3.6182800000000004,3.04296},   {0.21807341113346007,-0.647707935025115},
              {5.123,-6.11},{-1.05914748,4.42519664},       {-0.14942307391746978,-5.208155953378981} };

    LUDecomp<complex<double>> LU=LUconstruct_cpu(X);

    int flag=diff(LU.A,A_exact,1e-13);

    if(flag==0) cout<<"PASSED! LUconstruct_cpu passed complex double test!"<<endl;
    else cout<<"WARNING!!!!!!!!! LUconstruct_cpu failed complex double test!"<<endl;
}



void Tensor_2d_bl_cpu_test()
{
    int rank=0;
#ifdef MPI_HAO
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if(rank==0)
    {
        gmm_cpu_float_test();
        gmm_cpu_double_test();
        gmm_cpu_complex_float_test();
        gmm_cpu_complex_double_test();
        eigen_cpu_double_test();
        eigen_cpu_complex_double_test();
        LUconstruct_cpu_test();
    }

}

