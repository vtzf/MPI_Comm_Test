#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <complex.h>


void alltoallint(
    MPI_Comm comm, int nmodes, int nbands, int nk_a, int LCOMM
)
{
    int ierr = 0;
    int nprocs = 0;
    int myid = 0;

    ierr = MPI_Comm_size(comm,&nprocs);
    ierr = MPI_Comm_rank(comm,&myid);

    int i, j;
    int count = nk_a/nprocs;
    int nk_a1 = count*nprocs;

    int Len = nmodes*nbands*nbands*count;
    int scount = Len*count;

    int * epc_sbuf = (int*)calloc((nk_a1*Len),sizeof(int));
    int * epc_rbuf = (int*)malloc((nk_a1*Len)*sizeof(int));

    if (LCOMM != 0)
    {
        MPI_Alltoall(
            &epc_sbuf[0],scount,
            MPI_INT,epc_rbuf,scount,
            MPI_INT,comm
        );
    }
    free(epc_sbuf); free(epc_rbuf);
}


void alltoallcplx(
    MPI_Comm comm, int nmodes, int nbands, int nk_a, int LCOMM
)
{
    int ierr = 0;
    int nprocs = 0;
    int myid = 0;

    ierr = MPI_Comm_size(comm,&nprocs);
    ierr = MPI_Comm_rank(comm,&myid);

    int i, j;
    int count = nk_a/nprocs;
    int nk_a1 = count*nprocs;

    int Len = nmodes*nbands*nbands*count;
    int scount = Len*count;

    double complex * epc_sbuf = (double complex*)calloc(\
                                (nk_a1*Len),sizeof(double complex));
    double complex * epc_rbuf = (double complex*)malloc(\
                                (nk_a1*Len)*sizeof(double complex));

    if (LCOMM != 0)
    {
        MPI_Alltoall(
            &epc_sbuf[0],scount,
            MPI_DOUBLE_COMPLEX,epc_rbuf,scount,
            MPI_DOUBLE_COMPLEX,comm
        );
    }
    free(epc_sbuf); free(epc_rbuf);
}


void alltoallvint(
    MPI_Comm comm, int nmodes, int nbands, int nk_a, int LCOMM
)
{
    int ierr = 0;
    int nprocs = 0;
    int myid = 0;

    ierr = MPI_Comm_size(comm,&nprocs);
    ierr = MPI_Comm_rank(comm,&myid);

    int i, j;
    int * k_proc = (int*)malloc((nprocs+1)*sizeof(int));
    int * k_proc_num = (int*)malloc((nprocs+1)*sizeof(int));

    k_proc[0] = 0;
    for (i=0;i<nprocs;i++)
    {
        k_proc[i+1] = ((i+1)*nk_a)/nprocs;
        k_proc_num[i] = k_proc[i+1] - k_proc[i];
    }

    int nk_p = k_proc_num[myid];
    int Len = nmodes*nbands*nbands*nk_p;

    int * scount = (int*)malloc((nprocs)*sizeof(int));
    int * sdispl = (int*)malloc((nprocs)*sizeof(int));
    int * rcount = (int*)malloc((nprocs)*sizeof(int));
    int * rdispl = (int*)malloc((nprocs)*sizeof(int));
    int * epc_sbuf = (int*)calloc((nk_a*Len),sizeof(int));
    int * epc_rbuf = (int*)malloc((nk_a*Len)*sizeof(int));
    
    for (j=0;j<nprocs;j++)
    {
        scount[j] = Len*k_proc_num[j];
        rcount[j] = Len*k_proc_num[j];
        sdispl[j] = Len*k_proc[j];
        rdispl[j] = Len*k_proc[j];
    }
    if (LCOMM != 0)
    {
        MPI_Alltoallv(
            &epc_sbuf[0],&scount[0],&sdispl[0],
            MPI_INT,epc_rbuf,&rcount[0],&rdispl[0],
            MPI_INT,comm
        );
    }
    free(epc_sbuf); free(epc_rbuf); free(scount); free(rcount);
    free(sdispl); free(rdispl); free(k_proc); free(k_proc_num);
}


void alltoallvcplx(
    MPI_Comm comm, int nmodes, int nbands, int nk_a, int LCOMM
)
{
    int ierr = 0;
    int nprocs = 0;
    int myid = 0;

    ierr = MPI_Comm_size(comm,&nprocs);
    ierr = MPI_Comm_rank(comm,&myid);

    int i, j;
    int * k_proc = (int*)malloc((nprocs+1)*sizeof(int));
    int * k_proc_num = (int*)malloc((nprocs+1)*sizeof(int));

    k_proc[0] = 0;
    for (i=0;i<nprocs;i++)
    {
        k_proc[i+1] = ((i+1)*nk_a)/nprocs;
        k_proc_num[i] = k_proc[i+1] - k_proc[i];
    }

    int nk_p = k_proc_num[myid];
    int Len = nmodes*nbands*nbands*nk_p;

    int * scount = (int*)malloc((nprocs)*sizeof(int));
    int * sdispl = (int*)malloc((nprocs)*sizeof(int));
    int * rcount = (int*)malloc((nprocs)*sizeof(int));
    int * rdispl = (int*)malloc((nprocs)*sizeof(int));
    double complex * epc_sbuf = (double complex*)calloc(\
                                (nk_a*Len),sizeof(double complex));
    double complex * epc_rbuf = (double complex*)malloc(\
                                (nk_a*Len)*sizeof(double complex));
    
    for (j=0;j<nprocs;j++)
    {
        scount[j] = Len*k_proc_num[j];
        rcount[j] = Len*k_proc_num[j];
        sdispl[j] = Len*k_proc[j];
        rdispl[j] = Len*k_proc[j];
    }
    if (LCOMM != 0)
    {
        MPI_Alltoallv(
            &epc_sbuf[0],&scount[0],&sdispl[0],
            MPI_DOUBLE_COMPLEX,epc_rbuf,&rcount[0],&rdispl[0],
            MPI_DOUBLE_COMPLEX,comm
        );
    }
    free(epc_sbuf); free(epc_rbuf); free(scount); free(rcount);
    free(sdispl); free(rdispl); free(k_proc); free(k_proc_num);
}


int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
  
    int h, i, j;  
    int ierr = 0;
    int myid = 0;
    MPI_Comm comm = MPI_COMM_WORLD;
    ierr = MPI_Comm_rank(comm,&myid);

    double starttime0, endtime0;
    double starttime1, endtime1;
    double starttime2, endtime2;
    double starttime3, endtime3;

    for (i=1024;i<=32768;i*=2)
    {
        starttime0 = 0.0;
        endtime0 = 0.0;
        starttime1 = 0.0;
        endtime1 = 0.0;
        starttime2 = 0.0;
        endtime2 = 0.0;
        starttime3 = 0.0;
        endtime3 = 0.0;
        for (h=0;h<10;h++)
        {
            starttime0 += MPI_Wtime();
            alltoallcplx(comm,1,1,i,1);
            endtime0 += MPI_Wtime();
            starttime1 += MPI_Wtime();
            alltoallint(comm,1,1,i*2,1);
            endtime1 += MPI_Wtime();
            starttime2 += MPI_Wtime();
            alltoallvcplx(comm,1,1,i,1);
            endtime2 += MPI_Wtime();
            starttime3 += MPI_Wtime();
            alltoallvint(comm,1,1,i*2,1);
            endtime3 += MPI_Wtime();
        }
        if (myid==0)
        {
            j = i*i/(1024*1024)*16;
            printf("Rank %d, alltoallcplx  with %5dMB data: %.6fs.\n",myid,j,(endtime0-starttime0)/10);
            printf("Rank %d, alltoallint   with %5dMB data: %.6fs.\n",myid,j,(endtime1-starttime1)/10);
            printf("Rank %d, alltoallvcplx with %5dMB data: %.6fs.\n",myid,j,(endtime2-starttime2)/10);
            printf("Rank %d, alltoallvint  with %5dMB data: %.6fs.\n",myid,j,(endtime3-starttime3)/10);
        }
    }

    MPI_Finalize();
    return 0;
}
