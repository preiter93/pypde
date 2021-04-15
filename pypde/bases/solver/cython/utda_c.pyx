import numpy as np
cimport cython
cimport numpy as np

ctypedef np.float64_t real_t

@cython.boundscheck(False)
@cython.wraparound(False)
def solve_triangular_c(real_t[:, ::1] R,
               real_t[::1] b):
               #real_t[::1] x):
    cdef:
        unsigned int n = b.shape[0]
        unsigned int i,j,k
        cdef np.ndarray[real_t,ndim=1] x = np.empty((b.shape[0]),dtype=np.float64)

    for j in range(n):
        i = n-1-j
        x[i] = b[i]/R[i,i]
        if  i!=0: 
            for k in range(i):
                b[k] -= x[i]*R[k,i]
    return x