#ifdef MPI_HAO
#include <mpi.h>
#endif
#include <cmath>
#include "tensor_element_wise.h"

using namespace std;
using namespace tensor_hao;

void tensor_conj_test()
{
    Tensor_hao< complex<double>,2>  tensor_a(3,2);
    tensor_a={ {1.0 ,2.0} , {3.0 ,4.0} , {5.0 ,6.0} , {7.0 ,8.0} , {9.0 ,10.0} , {11.0 ,12.0} };

    Tensor_hao< complex<double>,2> tensor_b = conj( tensor_a );

    int flag=0; 
    for(int j=0; j<2; j++)
    {
        for(int i=0; i<3; i++)
        {
            if( std::abs( tensor_b(i,j)-conj( tensor_a(i,j) ) ) >1e-12 ) flag++;
        }
    }
    if(flag==0) cout<<"PASSED! Tensor conj passed complex<double> test!"<<endl;
    else cout<<"WARNING!!!!Tensor conj failed complex<double> test!"<<endl;

}

void tensor_exp_test()
{
    Tensor_hao<double,2>  tensor_a(3,4);
    tensor_a={1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

    Tensor_hao<double,2> tensor_b = exp( tensor_a );

    int flag=0;
    for(int j=0; j<4; j++)
    {
        for(int i=0; i<3; i++)
        {
            if( std::abs( tensor_b(i,j)-exp( tensor_a(i,j) ) ) >1e-12 ) flag++;
        }
    }
    if(flag==0) cout<<"PASSED! Tensor exp passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor exp failed double test!"<<endl;

}

void tensor_minus_test()
{
    Tensor_hao<double,2>  tensor_a(3,4);
    tensor_a={1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

    Tensor_hao<double,2> tensor_b = -tensor_a;

    int flag=0;
    for(int j=0; j<4; j++)
    {
        for(int i=0; i<3; i++)
        {
            if( std::abs( tensor_b(i,j) + tensor_a(i,j) ) >1e-12 ) flag++;
        }
    }
    if(flag==0) cout<<"PASSED! Tensor minus passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor minus failed double test!"<<endl;

}

void Tensor_add_Tensor_test()
{
    Tensor_hao<double,3>  tensor_a(3,4,5);
    int L = tensor_a.size();
    double* p_a = tensor_a.data();
    for(int i=0; i<L; i++) p_a[i] = i*2.0;

    Tensor_hao<double,3>  tensor_b(3,4,5);
    double* p_b = tensor_b.data();
    for(int i=0; i<L; i++) p_b[i] = i*1.0;

    //Tensor_hao_ref<double,3> tensor_a_ref = tensor_a;
    //Tensor_hao_ref<double,3> tensor_b_ref = tensor_b;
    Tensor_hao<double,3> tensor_c = tensor_a + tensor_b;
    double* p_c = tensor_c.data();

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p_c[i]-i*3.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor + Tensor passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor + Tensor failed double test!"<<endl;
}

void Tensor_minus_Tensor_test()
{
    Tensor_hao<double,3>  tensor_a(3,4,5);
    int L = tensor_a.size();
    double* p_a = tensor_a.data();
    for(int i=0; i<L; i++) p_a[i] = i*2.0;

    Tensor_hao<double,3>  tensor_b(3,4,5);
    double* p_b = tensor_b.data();
    for(int i=0; i<L; i++) p_b[i] = i*1.0;

    //Tensor_hao_ref<double,3> tensor_a_ref = tensor_a;
    //Tensor_hao_ref<double,3> tensor_b_ref = tensor_b;
    Tensor_hao<double,3> tensor_c = tensor_a - tensor_b;
    double* p_c = tensor_c.data();

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p_c[i]-i*1.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor - Tensor passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor - Tensor failed double test!"<<endl;
}

void Tensor_time_Tensor_test()
{
    Tensor_hao<double,3>  tensor_a(3,4,5);
    int L = tensor_a.size();
    double* p_a = tensor_a.data();
    for(int i=0; i<L; i++) p_a[i] = i*2.0;

    Tensor_hao<double,3>  tensor_b(3,4,5);
    double* p_b = tensor_b.data();
    for(int i=0; i<L; i++) p_b[i] = i*1.0;

    //Tensor_hao_ref<double,3> tensor_a_ref = tensor_a;
    //Tensor_hao_ref<double,3> tensor_b_ref = tensor_b;
    Tensor_hao<double,3> tensor_c = tensor_a * tensor_b;
    double* p_c = tensor_c.data();

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p_c[i]-i*i*2.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor * Tensor passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor * Tensor failed double test!"<<endl;
}

void Tensor_divide_Tensor_test()
{
    Tensor_hao<double,3>  tensor_a(3,4,5);
    int L = tensor_a.size();
    double* p_a = tensor_a.data();
    for(int i=0; i<L; i++) p_a[i] = i*2.0;

    Tensor_hao<double,3>  tensor_b(3,4,5);
    double* p_b = tensor_b.data();
    for(int i=0; i<L; i++) p_b[i] = i*1.0;

    //Tensor_hao_ref<double,3> tensor_a_ref = tensor_a;
    //Tensor_hao_ref<double,3> tensor_b_ref = tensor_b;
    Tensor_hao<double,3> tensor_c = tensor_a / tensor_b;
    double* p_c = tensor_c.data();

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p_c[i]-2.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor / Tensor passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor / Tensor failed double test!"<<endl;
}

void Tensor_T_add_test()
{
    Tensor_hao<double,3>  tensor_a(3,4,5);
    int L = tensor_a.size();
    double* p_a = tensor_a.data();
    for(int i=0; i<L; i++) p_a[i] = i*2.0;

    double b =3.0;

    //Tensor_hao_ref<double,3> tensor_a_ref = tensor_a;
    Tensor_hao<double,3> tensor_c = tensor_a + b ;
    double* p_c = tensor_c.data();

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p_c[i]-i*2.0-3.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor T + passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor T + failed double test!"<<endl;
}

void Tensor_T_minus_test()
{
    Tensor_hao<double,3>  tensor_a(3,4,5);
    int L = tensor_a.size();
    double* p_a = tensor_a.data();
    for(int i=0; i<L; i++) p_a[i] = i*2.0;

    double b =3.0;

    //Tensor_hao_ref<double,3> tensor_a_ref = tensor_a;
    Tensor_hao<double,3> tensor_c = tensor_a -b;
    double* p_c = tensor_c.data();

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p_c[i]-i*2.0+3.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor T - passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor T - failed double test!"<<endl;
}

void Tensor_T_time_test()
{
    Tensor_hao<double,3>  tensor_a(3,4,5);
    int L = tensor_a.size();
    double* p_a = tensor_a.data();
    for(int i=0; i<L; i++) p_a[i] = i*2.0;

    double b =3.0;

    //Tensor_hao_ref<double,3> tensor_a_ref = tensor_a;
    Tensor_hao<double,3> tensor_c = tensor_a * b  ;
    double* p_c = tensor_c.data();

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p_c[i]-i*2.0*3.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor T * passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor T * failed double test!"<<endl;
}

void Tensor_T_div_test()
{
    Tensor_hao<double,3>  tensor_a(3,4,5);
    int L = tensor_a.size();
    double* p_a = tensor_a.data();
    for(int i=0; i<L; i++) p_a[i] = i*2.0;

    double b =3.0;

    //Tensor_hao_ref<double,3> tensor_a_ref = tensor_a;
    Tensor_hao<double,3> tensor_c = tensor_a / b ;
    double* p_c = tensor_c.data();

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p_c[i]-i*2.0/3.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor T / passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor T / failed double test!"<<endl;
}

void Tensor_diff_test()
{
    Tensor_hao<double,2>  tensor_a(3,4);
    Tensor_hao<double,2>  tensor_b(3,4);
    Tensor_hao_ref<double,2>  tensor_b_ref(tensor_b);
    tensor_a = {1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.};
    tensor_b = {1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.};
 
    int flag=diff(tensor_a, tensor_b_ref, 1e-12);

    if(flag==0) cout<<"PASSED! Tensor diff passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor diff failed double test!"<<endl;
}


void Tensor_element_wise_test()
{
    int rank=0;
#ifdef MPI_HAO
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if(rank==0)
    {
        tensor_conj_test();
        tensor_exp_test();
        tensor_minus_test();
        Tensor_add_Tensor_test();
        Tensor_minus_Tensor_test();
        Tensor_time_Tensor_test();
        Tensor_divide_Tensor_test();
        Tensor_T_add_test();
        Tensor_T_minus_test();
        Tensor_T_time_test();
        Tensor_T_div_test();
        Tensor_diff_test();
        cout<<endl;
    }
}
