import numpy as np 
from scipy.sparse import csr_matrix, csc_matrix 

def tosparse(A,tol=1e-12,format="csc"):
    ''' 
    Sets elements of A which are smaller than tol to zero
    and returns a sparse version of A

    Input
        A: nd array

    Return
        sparse matrix
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

    Input
        a: 1d array
        b: Nd array

    Return
        Nd array
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

    Input
        a: 1d array
        b: Nd array

    Return
        Nd array
    '''
    assert a.shape[0] == b.shape[0], 'First dimension is different'
    assert b.ndim >= a.ndim, 'a has more dimensions than b'

    # add extra dimensions so that a will broadcast
    extra_dims = b.ndim - a.ndim
    newshape = a.shape + (1,)*extra_dims
    new_a = a.reshape(newshape)

    return new_a + b


def extract_diag(M,k=(-2,0,2)):
    '''
    Extracts diagonals from Matrix M

    Input
        M: Matrix
        k: tuple
            contains offset of diagonals

    Return
        tuple of arrays
    '''
    return tuple( [np.diag(M,i) for i in k] )


def zero_pad(array: np.ndarray, target_length: int, axis: int = 0):
    '''
    Example:
    a = np.array([ [ 1.,  1.],
                   [ 1.,  1.] ])

    zero_pad(a,3,0)

    > array([[1., 1.],
       [1., 1.],
       [0., 0.]])
    '''

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)