#ifdef MPI_HAO
#include <mpi.h>
#endif
#include <cmath>
#include "tensor_hao_ref.h"
#include "tensor_hao.h"

using namespace std;
using namespace tensor_hao;

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

    if(flag==0) cout<<"Tensor_core read passed double 1 test!"<<endl;
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

    if(flag==0) cout<<"Tensor_core write passed double 1 test!"<<endl;
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

    if(flag==0) cout<<"Tensor_core read passed double 2 test!"<<endl;
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

    if(flag==0) cout<<"Tensor_core write passed double 2 test!"<<endl;
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

    if(flag==0) cout<<"Tensor_core read passed double 3 test!"<<endl;
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

    if(flag==0) cout<<"Tensor_core write passed double 3 test!"<<endl;
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

    if(flag==0) cout<<"Tensor_core read passed double 5 test!"<<endl;
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

    if(flag==0) cout<<"Tensor_core write passed double 5 test!"<<endl;
    else cout<<"WARNING!!!!Tensor_core write failed double 5 test!"<<endl;
}


void Tensor_core_test()
{
    int rank=0;
#ifdef MPI_HAO
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    if(rank==0)
    {
        Tensor_core_read_1_test();
        Tensor_core_write_1_test();
        Tensor_core_read_2_test();
        Tensor_core_write_2_test();
        Tensor_core_read_3_test();
        Tensor_core_write_3_test();
        Tensor_core_read_5_test();
        Tensor_core_write_5_test();
    }

}
