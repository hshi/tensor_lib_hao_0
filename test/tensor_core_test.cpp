#ifdef MPI_HAO
#include <mpi.h>
#endif
#include <cmath>
#include <complex>
#include "tensor_hao_ref.h"
#include "tensor_hao.h"

using namespace std;
using namespace tensor_hao;

void Tensor_core_nptr_test()
{
    Tensor_hao<double,5>  tensor(2,3,4,5,6);
    const int* n_ptr = tensor.n_ptr();
    int flag=0;
    for(int i=0; i<5; i++)
    {
        if( n_ptr[i] != (i+2) ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core passed n_ptr test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core failed n_ptr test!"<<endl;    
}

void Tensor_core_read_1_test()
{
    Tensor_hao<double,1>  tensor(3);
    int L = tensor.size();
    double* p = tensor.data();

    for(int i=0; i<L; i++) p[i] = i*1.0;

    int flag=0; int count=0;
    for(int k=0; k<3; k++)
    {
        if( std::abs( p[count]-tensor(k) ) > 1e-12 ) flag++;
        count++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core read passed double 1 test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core read failed double 1 test!"<<endl;
}


void Tensor_core_write_1_test()
{
    Tensor_hao<double,1>  tensor(3);
    int L = tensor.size();
    double* p = tensor.data();
    int flag=0; int count=0;

    count=0;
    for(int k=0; k<3; k++) 
    {
        tensor(k) = count*1.0;
        count++;
    }

    count=0;
    for(int i=0; i<L; i++) 
    {
        if( std::abs( p[count]-count ) > 1e-12 ) flag++;
        count++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core write passed double 1 test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core write failed double 1 test!"<<endl;
}



void Tensor_core_read_2_test()
{
    Tensor_hao<double,2>  tensor(3,4);
    int L = tensor.size();
    double* p = tensor.data();

    for(int i=0; i<L; i++) p[i] = i*1.0;

    int flag=0; int count=0;
    for(int j=0; j<4; j++)
    {
        for(int k=0; k<3; k++)
        {
            if( std::abs( p[count]-tensor(k,j) ) > 1e-12 ) flag++;
            count++;
        }
    }

    if(flag==0) cout<<"PASSED! Tensor_core read passed double 2 test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core read failed double 2 test!"<<endl;
}

void Tensor_core_write_2_test()
{
    Tensor_hao<double,2>  tensor(3,4);
    int L = tensor.size();
    double* p = tensor.data();
    int flag=0; int count=0;

    count=0;
    for(int j=0; j<4; j++)
    {
        for(int k=0; k<3; k++)
        {
            tensor(k,j) = count*1.0;
            count++;
        }
    }

    count=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p[count]-count ) > 1e-12 ) flag++;
        count++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core write passed double 2 test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core write failed double 2 test!"<<endl;
}


void Tensor_core_read_3_test()
{
    Tensor_hao<double,3>  tensor(3,4,5);
    int L = tensor.size();
    double* p = tensor.data();

    for(int i=0; i<L; i++) p[i] = i*1.0;

    int flag=0; int count=0;
    for(int i=0; i<5; i++)
    {
        for(int j=0; j<4; j++)
        {
            for(int k=0; k<3; k++) 
            {
                if( std::abs( p[count]-tensor(k,j,i) ) > 1e-12 ) flag++;
                count++;
            }
        }
    }

    if(flag==0) cout<<"PASSED! Tensor_core read passed double 3 test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core read failed double 3 test!"<<endl;
}

void Tensor_core_write_3_test()
{
    Tensor_hao<double,3>  tensor(3,4,5);
    int L = tensor.size();
    double* p = tensor.data();
    int flag=0; int count=0;

    count=0;
    for(int i=0; i<5; i++)
    {
        for(int j=0; j<4; j++)
        {
            for(int k=0; k<3; k++)
            {
                tensor(k,j,i)=count*1.0;
                count++;
            }
        }
    }

    count=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p[count]-count ) > 1e-12 ) flag++;
        count++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core write passed double 3 test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core write failed double 3 test!"<<endl;
}


void Tensor_core_read_5_test()
{
    Tensor_hao<double,5>  tensor(3,4,5,6,7);
    int L = tensor.size();
    double* p = tensor.data();

    for(int i=0; i<L; i++) p[i] = i*1.0;

    int flag=0; int count=0;
    for(int i5=0; i5<7; i5++)
    {
        for(int i4=0; i4<6; i4++)
        {
            for(int i=0; i<5; i++)
            {
                for(int j=0; j<4; j++)
                {
                    for(int k=0; k<3; k++)
                    {
                        if( std::abs( p[count]-tensor(k,j,i,i4,i5) ) > 1e-12 ) flag++;
                        count++;
                    }
                }
            }
        }
    }

    if(flag==0) cout<<"PASSED! Tensor_core read passed double 5 test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core read failed double 5 test!"<<endl;
}

void Tensor_core_write_5_test()
{
    Tensor_hao<double,5>  tensor(3,4,5,6,7);
    int L = tensor.size();
    double* p = tensor.data();
    int flag=0; int count=0;

    count=0;
    for(int i5=0; i5<7; i5++)
    {
        for(int i4=0; i4<6; i4++)
        {
            for(int i=0; i<5; i++)
            {
                for(int j=0; j<4; j++)
                {
                    for(int k=0; k<3; k++)
                    {
                        tensor(k,j,i,i4,i5) = count*1.0;
                        count++;
                    }
                }
            }
        }
    }

    count=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p[count]-count ) > 1e-12 ) flag++;
        count++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core write passed double 5 test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core write failed double 5 test!"<<endl;
}


void Tensor_core_add_equal_Tensor_core_test()
{
    Tensor_hao<double,3>  tensor(3,4,5);
    int L = tensor.size();
    double* p = tensor.data();
    for(int i=0; i<L; i++) p[i] = i*2.0;

    Tensor_hao<double,3>  tensor_b(3,4,5);
    double* p_b = tensor_b.data();
    for(int i=0; i<L; i++) p_b[i] = i*1.0;

    tensor+=tensor_b;

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p[i]-i*3.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core += Tensor_core passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core += Tensor_core failed double test!"<<endl;
}


void Tensor_core_minus_equal_Tensor_core_test()
{
    Tensor_hao<double,3>  tensor(3,4,5);
    int L = tensor.size();
    double* p = tensor.data();
    for(int i=0; i<L; i++) p[i] = i*3.0;

    Tensor_hao<double,3>  tensor_b(3,4,5);
    double* p_b = tensor_b.data();
    for(int i=0; i<L; i++) p_b[i] = i*1.0;

    tensor-=tensor_b;

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p[i]-i*2.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core -= Tensor_core passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core -= Tensor_core failed double test!"<<endl;
}


void Tensor_core_min_add_equal_Tensor_core_test()
{
    Tensor_hao<double,3>  tensor(3,4,5);
    int L = tensor.size();
    double* p = tensor.data();
    for(int i=0; i<L; i++) p[i] = i*3.0;

    Tensor_hao<double,3>  tensor_b(3,4,5);
    double* p_b = tensor_b.data();
    for(int i=0; i<L; i++) p_b[i] = i*1.0;

    tensor.min_add_equal(tensor_b);

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p[i]+i*2.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core = -Tensor_core + Tensor_core_b passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core = -Tensor_core + Tensor_core_b failed double test!"<<endl;
}


void Tensor_core_time_equal_Tensor_core_test()
{
    Tensor_hao<double,3>  tensor(3,4,5);
    int L = tensor.size();
    double* p = tensor.data();
    for(int i=0; i<L; i++) p[i] = i*3.0;

    Tensor_hao<double,3>  tensor_b(3,4,5);
    double* p_b = tensor_b.data();
    for(int i=0; i<L; i++) p_b[i] = i*1.0;

    tensor*=tensor_b;

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p[i]-i*i*3.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core *= Tensor_core passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core *= Tensor_core failed double test!"<<endl;
}

void Tensor_core_divide_equal_Tensor_core_test()
{
    Tensor_hao<double,3>  tensor(3,4,5);
    int L = tensor.size();
    double* p = tensor.data();
    for(int i=0; i<L; i++) p[i] = i*3.0;

    Tensor_hao<double,3>  tensor_b(3,4,5);
    double* p_b = tensor_b.data();
    for(int i=0; i<L; i++) p_b[i] = i*1.0;

    tensor/=tensor_b;

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p[i]-3.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core /= Tensor_core passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core /= Tensor_core failed double test!"<<endl;
}

void Tensor_core_inv_div_equal_Tensor_core_test()
{
    Tensor_hao<double,3>  tensor(3,4,5);
    int L = tensor.size();
    double* p = tensor.data();
    for(int i=0; i<L; i++) p[i] = i*3.0;

    Tensor_hao<double,3>  tensor_b(3,4,5);
    double* p_b = tensor_b.data();
    for(int i=0; i<L; i++) p_b[i] = i*1.0;

    tensor.inv_div_equal(tensor_b);

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p[i]-1/3.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core =  Tensor_core_b / Tensor_core passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core =  Tensor_core_b / Tensor_core failed double test!"<<endl;
}


void Tensor_core_add_equal_T_test()
{
    Tensor_hao<double,3>  tensor(3,4,5);
    int L = tensor.size();
    double* p = tensor.data();
    for(int i=0; i<L; i++) p[i] = i*2.0;

    double b=3.0;

    tensor+=b;

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p[i]-i*2.0-3.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core += T passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core += T failed double test!"<<endl;
}

void Tensor_core_minus_equal_T_test()
{
    Tensor_hao<double,3>  tensor(3,4,5);
    int L = tensor.size();
    double* p = tensor.data();
    for(int i=0; i<L; i++) p[i] = i*2.0;

    double b=3.0;

    tensor-=b;

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p[i]-i*2.0+3.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core -= T passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core -= T failed double test!"<<endl;
}

void Tensor_core_min_add_equal_T_test()
{
    Tensor_hao<double,3>  tensor(3,4,5);
    int L = tensor.size();
    double* p = tensor.data();
    for(int i=0; i<L; i++) p[i] = i*2.0;

    double b=3.0;

    tensor.min_add_equal(b);

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p[i]+i*2.0-3.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core = -Tensor_core + T passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core = -Tensor_core + T failed double test!"<<endl;
}


void Tensor_core_time_equal_T_test()
{
    Tensor_hao<double,3>  tensor(3,4,5);
    int L = tensor.size();
    double* p = tensor.data();
    for(int i=0; i<L; i++) p[i] = i*2.0;

    double b=3.0;

    tensor*=b;

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p[i]-i*2.0*3.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core *= T passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core *= T failed double test!"<<endl;
}

void Tensor_core_divide_equal_T_test()
{
    Tensor_hao<double,3>  tensor(3,4,5);
    int L = tensor.size();
    double* p = tensor.data();
    for(int i=0; i<L; i++) p[i] = i*2.0;

    double b=3.0;

    tensor/=b;

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p[i]-i*2.0/3.0 ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core /= T passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core /= T failed double test!"<<endl;
}


void Tensor_core_inv_div_equal_T_test()
{
    Tensor_hao<double,3>  tensor(3,4,5);
    int L = tensor.size();
    double* p = tensor.data();
    for(int i=0; i<L; i++) p[i] = i*2.0;

    double b=3.0;

    tensor.inv_div_equal(b);

    int flag=0;
    for(int i=0; i<L; i++)
    {
        if( std::abs( p[i]-3.0/(2.0*i) ) > 1e-12 ) flag++;
    }

    if(flag==0) cout<<"PASSED! Tensor_core =  T / Tensor_core passed double test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core =  T / Tensor_core failed double test!"<<endl;
}

void Tensor_core_sum_test()
{
    Tensor_hao< complex<double>,2>  tensor_a(3,2);
    tensor_a={ {1.0 ,2.0} , {3.0 ,4.0} , {5.0 ,6.0} , {7.0 ,8.0} , {9.0 ,10.0} , {11.0 ,12.0} };

    complex<double> exact(36., 42.0);

    if( abs( tensor_a.sum() - exact ) < 1e-12 )  cout<<"PASSED! Tensor_core sum passed complex double test!"<<endl; 
    else  cout<<"WARNING!!!! Tensor_core sum failed complex double test!"<<endl; 
}

void Tensor_core_mean_test()
{
    Tensor_hao< complex<double>,2>  tensor_a(3,2);
    tensor_a={ {1.0 ,2.0} , {3.0 ,4.0} , {5.0 ,6.0} , {7.0 ,8.0} , {9.0 ,10.0} , {11.0 ,12.0} };

    complex<double> exact(6., 7.0);

    if( abs( tensor_a.mean() - exact ) < 1e-12 )  cout<<"PASSED! Tensor_core mean passed complex double test!"<<endl;
    else  cout<<"WARNING!!!! Tensor_core mean failed complex double test!"<<endl; 
}

void Tensor_core_test()
{
    int rank=0;
#ifdef MPI_HAO
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if(rank==0)
    {
        Tensor_core_nptr_test();
        Tensor_core_read_1_test();
        Tensor_core_write_1_test();
        Tensor_core_read_2_test();
        Tensor_core_write_2_test();
        Tensor_core_read_3_test();
        Tensor_core_write_3_test();
        Tensor_core_read_5_test();
        Tensor_core_write_5_test();
        Tensor_core_add_equal_Tensor_core_test();
        Tensor_core_minus_equal_Tensor_core_test();
        Tensor_core_min_add_equal_Tensor_core_test();
        Tensor_core_time_equal_Tensor_core_test();
        Tensor_core_divide_equal_Tensor_core_test();
        Tensor_core_inv_div_equal_Tensor_core_test();
        Tensor_core_add_equal_T_test();
        Tensor_core_minus_equal_T_test();
        Tensor_core_min_add_equal_T_test(); 
        Tensor_core_time_equal_T_test();
        Tensor_core_divide_equal_T_test();
        Tensor_core_inv_div_equal_T_test();
        Tensor_core_sum_test();
        Tensor_core_mean_test();
        cout<<endl;
    }

}
