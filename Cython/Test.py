import alltoall
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
myid = comm.Get_rank()
nprocs = comm.Get_size()

for i in range(10,16):
    j = 2**i
    starttime0 = 0.0
    endtime0 = 0.0
    starttime1 = 0.0
    endtime1 = 0.0
    starttime2 = 0.0
    endtime2 = 0.0
    starttime3 = 0.0
    endtime3 = 0.0
    for h in range(10):
        starttime0 += MPI.Wtime()
        alltoall.alltoallcplx(comm,1,1,j,True)
        endtime0 += MPI.Wtime()
        starttime1 += MPI.Wtime()
        alltoall.alltoallint(comm,1,1,j*2,True)
        endtime1 += MPI.Wtime()
        starttime2 += MPI.Wtime()
        alltoall.alltoallvcplx(comm,1,1,j,True)
        endtime2 += MPI.Wtime()
        starttime3 += MPI.Wtime()
        alltoall.alltoallvint(comm,1,1,j*2,True)
        endtime3 += MPI.Wtime()

    if myid == 0:
        k = j*j/(1024*1024)*16
        print("Rank %d, alltoallcplx  with %5dMB data: %.6fs."%(myid,k,(endtime0-starttime0)/10))
        print("Rank %d, alltoallint   with %5dMB data: %.6fs."%(myid,k,(endtime1-starttime1)/10))
        print("Rank %d, alltoallvcplx with %5dMB data: %.6fs."%(myid,k,(endtime2-starttime2)/10))
        print("Rank %d, alltoallvint  with %5dMB data: %.6fs."%(myid,k,(endtime3-starttime3)/10))
