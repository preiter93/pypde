import numpy as np 

def eigen_decomp(A):
    ''' 
    Eigendecomposition of A

    Input
        A: square Matrix

    Output
        w: Eigenvalues
        Q: Eigenvectors (columnwise)
        Qi: Inverse of Q
    '''
    w, Q = np.linalg.eig(A)
    argsort = np.argsort(w)
    w = w[argsort]
    Q = Q[:,argsort]
    Qi = np.linalg.inv(Q)
    return w,Q,Qi