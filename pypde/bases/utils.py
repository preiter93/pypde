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

def product(a, b):
    '''Product across the first dimension of b.

    Assumes a is 1-dimensional.
    Raises AssertionError if a.ndim > b.ndim or
     - the first dimensions are different
    '''
    assert a.shape[0] == b.shape[0], 'First dimension is different'
    assert b.ndim >= a.ndim, 'a has more dimensions than b'

    # add extra dimensions so that a will broadcast
    extra_dims = b.ndim - a.ndim
    newshape = a.shape + (1,)*extra_dims
    new_a = a.reshape(newshape)

    return new_a * b

def add(a, b):
    '''Summation across the first dimension of b.

    Assumes a is 1-dimensional.
    Raises AssertionError if a.ndim > b.ndim or
     - the first dimensions are different
    '''
    assert a.shape[0] == b.shape[0], 'First dimension is different'
    assert b.ndim >= a.ndim, 'a has more dimensions than b'

    # add extra dimensions so that a will broadcast
    extra_dims = b.ndim - a.ndim
    newshape = a.shape + (1,)*extra_dims
    new_a = a.reshape(newshape)

    return new_a + b