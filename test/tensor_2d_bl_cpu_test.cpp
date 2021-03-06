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

    int flag=0;

    LUDecomp<complex<double>> LU=LUconstruct_cpu(X);
    flag+=diff(LU.A,A_exact,1e-13);

    LUDecomp<complex<double>> LU_p=LUconstruct_cpu(move(X));
    flag+=diff(LU_p.A,A_exact,1e-13);

    if(flag==0) cout<<"PASSED! LUconstruct_cpu passed complex double test!"<<endl;
    else cout<<"WARNING!!!!!!!!! LUconstruct_cpu failed complex double test!"<<endl;
}

void inverse_cpu_test()
{
    Tensor_hao<complex<double>,2> A(3,3), INV_A, INV_A_exact(3,3);

    A = { {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
          {3.0,-2.0},   {2.0,0.0},    {5.123,3.11},
          {2.123,-5.11},{5.123,-6.11},{3,0.0} };

    INV_A_exact = { {-0.31516333912326305,0.13336022037456957} ,
                    {0.16746685439563874,-0.0779491606298965},
                    {-0.005504176768078849,0.1918486231848867},
                    {0.1412286826747599,-0.11408929794801193},
                    {-0.1402834127458906,0.038283792754219295},
                    {0.061029436341995695,0.01438130659499342},
                    {-0.01293596267860185,-0.1487405620815458},
                    {0.17584867623524927,-0.010672609392757534},
                    {-0.12306156095719788,-0.04540218264765162} };

    int flag=0;

    INV_A=inverse_cpu( LUconstruct_cpu(A) );
    flag+=diff(INV_A,INV_A_exact,1e-13);

    LUDecomp<complex<double>> x=LUconstruct_cpu(A);
    INV_A=inverse_cpu(x);
    flag+=diff(INV_A,INV_A_exact,1e-13);

    if(flag==0) cout<<"PASSED! Inverse_cpu passed complex double test!"<<endl;
    else cout<<"WARNING!!!!!!!!! Inverse_cpu failed complex double test!"<<endl;
}

void solve_lineq_cpu_test()
{
    int flag=0;

    Tensor_hao<complex<double>,2> A(3,3), B(3,2), X_exact(3,2), X; 

    A = { {1.0,0.0} ,   {3.0,4.0},    {2.123,3.11},
          {3.0,-2.0},   {2.0,0.0},    {5.123,3.11},
          {2.123,-5.11},{5.123,-6.11},{3,0.0}  };

    B = { {2.0,0.0} ,   {3.0,5.0},    {3.123,3.11},
          {3.0,-6.0},   {2.0,1.0},    {6.123,3.11} };

    X_exact = { {0.785989996146147, 0.12584834096778363} ,
                {0.3050317378766687,-0.22890518276854455},
                {-0.1429470443202702,0.20747587687923086},
                {0.6345942167676883, 1.253141477086266},
                {0.825768240961444,-0.8208234397212029},
                {0.6299516251873555,0.037643960766659545} };

    X=solve_lineq_cpu( LUconstruct_cpu(A), B );
    flag+=diff(X,X_exact,1e-13);

    X=solve_lineq_cpu( LUconstruct_cpu(A), move(B) );
    flag+=diff(X,X_exact,1e-13);

    if(flag==0) cout<<"PASSED! Solve_lineq_cpu passed complex double test!"<<endl;
    else cout<<"WARNING!!!!!!!!! Solve_lineq_cpu failed complex double test!"<<endl;

}

void QRMatrix_cpu_test()
{
    const int L0=3; const int L1=2;
    Tensor_hao<complex<double>,2> A(L0,L1), B(L0,L1), A_exact(L0,L1);
    Tensor_hao<double, 1> det_list(L1); 

    A = { {2.0,0.0} ,   {3.0,5.0},    {3.123,3.11},
          {3.0,-6.0},   {2.0,1.0},    {6.123,3.11} };

    B = A;

    A_exact = { {0.26392384387316437, 0} ,
                {0.3958857658097466 , 0.6598096096829109},
                {0.41211708220794624, 0.41040157722277065},
                {0.20568020122880237 , -0.7338652779407804},
                {-0.41851770493832796, -0.22064009932009565},
                {0.3071492824057953  ,0.3177382636670606} };

    double det=QRMatrix_cpu(A);

    double det_list_M = QRMatrix_cpu(B,det_list);

    double det_exact=51.76794728400964;

    int flag=0;
    for(int j=0; j<L1; j++)
    {
        for(int i=0; i<L0; i++) {if(abs(abs(A(i,j))-abs(A_exact(i,j)))>1e-12) flag++;} //Use abs for unexpected sign
    }

    if(abs(det-det_exact)>1e-12) flag++;

    for(int j=0; j<L1; j++)
    {
        for(int i=0; i<L0; i++) {if(abs(abs(B(i,j))-abs(A_exact(i,j)))>1e-12) flag++;} //Use abs for unexpected sign
    }

    if(abs(det-det_list_M)>1e-12) flag++;

    det_list_M=1.0; for(int i=0; i<L1; i++) det_list_M*=det_list(i);
    if(abs(det-det_list_M)>1e-12) flag++;

    if(flag==0) cout<<"PASSED! QRMatrix_cpu passed complex double test!"<<endl;
    else cout<<"WARNING!!!!!!!!! QRMatrix_cpu failed complex double test!"<<endl;
}

void SVDMatrix_cpu_test()
{
    const int L=3;
    Tensor_hao<complex<double>,2> U_before(L,L), U(L,L), V(L,L);
    Tensor_hao<double,1> D(L), D_exact(L);

    U_before = { {1.0,2.0}, {2.0,1.0}, {3.0,5.0},
                 {2.0,0.5}, {2.0,0.0}, {1.0,2.0},
                 {1.0,7.0}, {8.2,1.0}, {3.3,5.0} };

    D_exact = {14.003853260054132, 3.686182682653158, 1.2977484736955485};

    U=U_before;

    SVDMatrix_cpu(U, D, V);

    Tensor_hao<complex<double>,1> D_cd(L);
    Tensor_hao<complex<double>,2> U_back(L,L);
    for(int i=0; i<L; i++) D_cd(i)=D(i);

    gmm_cpu(U, D_Multi_Matrix(D_cd, V), U_back);

    int flag=0;
    flag+=diff(D,D_exact,1e-13);
    flag+=diff(U_before,U_back,1e-13);
    if(flag==0) cout<<"PASSED! SVDMatrix_cpu passed complex double test!"<<endl;
    else cout<<"WARNING!!!!!!!!! SVDMatrix_cpu failed complex double test!"<<endl;
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
        inverse_cpu_test();
        solve_lineq_cpu_test();
        QRMatrix_cpu_test();
        SVDMatrix_cpu_test();
        cout<<endl;
    }
}
