#cython: language_level=3
#cython: cdivision=True

cimport cython
import numpy as np
cimport numpy as np
from mpi4py cimport MPI
from mpi4py cimport libmpi as mpi
from libc.stdlib cimport malloc, free


@cython.boundscheck(False)
@cython.wraparound(False)
def alltoallint(
    MPI.Comm comm, int nmodes, int nbands, int nk_a, bint LCOMM
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef int ierr = 0
    cdef int nprocs = 0
    cdef int myid = 0

    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)

    cdef int i, j
    cdef int count = nk_a/nprocs
    cdef int nk_a1 = count*nprocs

    cdef int Len = nmodes*nbands*nbands*count
    cdef int scount = Len*count

    cdef int[:,:,:,:,::1] epc_sbuf = \
         np.zeros((nk_a1,count,nmodes,nbands,nbands),dtype=np.int32)
    cdef int * epc_rbuf

    epc_rbuf = <int*>malloc((nk_a1*Len)*sizeof(int))

    if LCOMM:
        mpi.MPI_Alltoall(
            &epc_sbuf[0,0,0,0,0],scount,
            mpi.MPI_INT,epc_rbuf,scount,
            mpi.MPI_INT,c_comm
        )

    free(epc_rbuf)


@cython.boundscheck(False)
@cython.wraparound(False)
def alltoallcplx(
    MPI.Comm comm, int nmodes, int nbands, int nk_a, bint LCOMM
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef int ierr = 0
    cdef int nprocs = 0
    cdef int myid = 0

    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)

    cdef int i, j
    cdef int count = nk_a/nprocs
    cdef int nk_a1 = count*nprocs

    cdef int Len = nmodes*nbands*nbands*count
    cdef int scount = Len*count

    cdef double complex[:,:,:,:,::1] epc_sbuf = \
         np.zeros((nk_a1,count,nmodes,nbands,nbands),dtype=np.complex128)
    cdef double complex * epc_rbuf

    epc_rbuf = <double complex*>malloc((nk_a1*Len)*sizeof(double complex))

    if LCOMM:
        mpi.MPI_Alltoall(
            &epc_sbuf[0,0,0,0,0],scount,
            mpi.MPI_DOUBLE_COMPLEX,epc_rbuf,scount,
            mpi.MPI_DOUBLE_COMPLEX,c_comm
        )

    free(epc_rbuf)


@cython.boundscheck(False)
@cython.wraparound(False)
def alltoallvint(
    MPI.Comm comm, int nmodes, int nbands, int nk_a, bint LCOMM
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef int ierr = 0
    cdef int nprocs = 0
    cdef int myid = 0

    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)

    cdef int i, j
    cdef int[::1] k_proc = np.zeros((nprocs+1),dtype=np.int32)
    cdef int[::1] k_proc_num = np.zeros((nprocs),dtype=np.int32)

    for i in range(nprocs):
        k_proc[i+1] = ((i+1)*nk_a)//nprocs
        k_proc_num[i] = k_proc[i+1]-k_proc[i]

    cdef int nk_p = k_proc_num[myid]
    cdef int Len = nmodes*nbands*nbands*nk_p

    cdef int[::1] scount = np.zeros((nprocs),dtype=np.int32)
    cdef int[::1] sdispl = np.zeros((nprocs),dtype=np.int32)
    cdef int[::1] rcount = np.zeros((nprocs),dtype=np.int32)
    cdef int[::1] rdispl = np.zeros((nprocs),dtype=np.int32)
    cdef int[:,:,:,:,::1] epc_sbuf = \
         np.zeros((nk_a,nk_p,nmodes,nbands,nbands),dtype=np.int32)
    cdef int * epc_rbuf

    epc_rbuf = <int*>malloc((nk_a*Len)*sizeof(int))

    for j in range(nprocs):
        scount[j] = Len*k_proc_num[j]
        rcount[j] = Len*k_proc_num[j]
        sdispl[j] = Len*k_proc[j]
        rdispl[j] = Len*k_proc[j]

    if LCOMM:
        mpi.MPI_Alltoallv(
            &epc_sbuf[0,0,0,0,0],&scount[0],&sdispl[0],
            mpi.MPI_INT,epc_rbuf,&rcount[0],&rdispl[0],
            mpi.MPI_INT,c_comm
        )

    free(epc_rbuf)


@cython.boundscheck(False)
@cython.wraparound(False)
def alltoallvcplx(
    MPI.Comm comm, int nmodes, int nbands, int nk_a, bint LCOMM
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef int ierr = 0
    cdef int nprocs = 0
    cdef int myid = 0

    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)

    cdef int i, j
    cdef int[::1] k_proc = np.zeros((nprocs+1),dtype=np.int32)
    cdef int[::1] k_proc_num = np.zeros((nprocs),dtype=np.int32)

    for i in range(nprocs):
        k_proc[i+1] = ((i+1)*nk_a)//nprocs
        k_proc_num[i] = k_proc[i+1]-k_proc[i]

    cdef int nk_p = k_proc_num[myid]
    cdef int Len = nmodes*nbands*nbands*nk_p

    cdef int[::1] scount = np.zeros((nprocs),dtype=np.int32)
    cdef int[::1] sdispl = np.zeros((nprocs),dtype=np.int32)
    cdef int[::1] rcount = np.zeros((nprocs),dtype=np.int32)
    cdef int[::1] rdispl = np.zeros((nprocs),dtype=np.int32)
    cdef double complex[:,:,:,:,::1] epc_sbuf = \
         np.zeros((nk_a,nk_p,nmodes,nbands,nbands),dtype=np.complex128)
    cdef double complex * epc_rbuf

    epc_rbuf = <double complex*>malloc((nk_a*Len)*sizeof(double complex))

    for j in range(nprocs):
        scount[j] = Len*k_proc_num[j]
        rcount[j] = Len*k_proc_num[j]
        sdispl[j] = Len*k_proc[j]
        rdispl[j] = Len*k_proc[j]

    if LCOMM:
        mpi.MPI_Alltoallv(
            &epc_sbuf[0,0,0,0,0],&scount[0],&sdispl[0],
            mpi.MPI_DOUBLE_COMPLEX,epc_rbuf,&rcount[0],&rdispl[0],
            mpi.MPI_DOUBLE_COMPLEX,c_comm
        )

    free(epc_rbuf)
    
