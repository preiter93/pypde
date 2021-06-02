import scipy.sparse as sp
import numpy as np


def tosparse(A, type="csr"):
    assert type in ["csr", "csc"], "to sparse implements type='csc/csr' "
    if type == "csr":
        return sp.csr_matrix(A)
    if type == "csc":
        return sp.csr_matrix(A)


def eigdecomp(A):
    """
    Eigendecomposition of A
    Input
        A: square Matrix
    Output
        w: Eigenvalues
        Q: Eigenvectors (columnwise)
        Qi: Inverse of Q
    """
    w, Q = np.linalg.eig(A)
    argsort = np.argsort(w)
    argsort = argsort[::-1]
    w = w[argsort]
    Q = Q[:, argsort]
    Qi = np.linalg.inv(Q)
    return w, Q, Qi
