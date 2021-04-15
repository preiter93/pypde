# cimport cython
# cimport numpy as np 
# import numpy as np 

# cpdef double[:] solve_twodma_c(double[:] d, double[:] u1, double[:] x):
#     ''' 
#     d: N
#         diagonal
#     u1: N-2
#         Diagonal with offset -2
#     x: array ndim==1
#         rhs
#     '''
#     cdef int i
#     cdef int n
#     cdef double[:] y = np.zeros((n))
#     n = d.shape[0]
#     x[0] = x[0]/d[0]
#     x[1] = x[1]/d[1]
#     for i in range(2,n - 1):
#         y[i] = (x[i] - u1[i-2]*y[i-2])/d[i]
#     return y

import numpy as np
cimport cython
cimport numpy as np

ctypedef np.float64_t real_t

def solve_twodma_c(real_t[::1] d,
               real_t[::1] u,
               real_t[::1] x):
    cdef:
        unsigned int n = x.shape[0]
        unsigned int i

    x[0] = x[0]/d[0]
    x[1] = x[1]/d[1]
    for i in range(2,n - 1):
        x[i] = (x[i] - u[i-2]*x[i-2])/d[i]
