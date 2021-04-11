import numpy as np 
from scipy.sparse import csr_matrix, csc_matrix 

def to_sparse(A,tol=1e-12,format="csc"):
    ''' 
    Sets elements of A which are smaller than tol to zero
    and returns a sparse version of A
    '''
    A[np.abs(A)<tol] = 0
    if format in "csc": 
        return csc_matrix(A)
    if format in "csr": 
        return csr_matrix(A)