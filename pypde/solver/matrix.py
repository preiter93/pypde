import numpy as np
import scipy.sparse as sp
from .memoize import memoized
from .utils import tosparse


class MatrixBase:
    """
    Store Scalars/Matrices and Information along which axis of
    the field they act

    Input
        M: scalar or ndarray
        axis: int
            Defines along which axis PM should be applied
        sparse: bool (optional)
            If true, matrix will be stored sparse
    """

    scalar = False

    def __init__(self, M, axis=0, sparse=False):
        assert isinstance(
            M, (float, int, np.ndarray, sp.csr_matrix, sp.csc_matrix)
        ), "Input must be a scalar or matrix (can be sparse)."

        self.axis = axis
        self.sparse = sparse

        # Input is a scalar
        if isinstance(M, float) or isinstance(M, int):
            self.value = float(M)
            self.scalar = True
        # Input is a Matrix
        else:
            self.value = M

        # Transform to sparse or array
        self.tosparse() if sparse else self.toarray()

    def dot(self, b):
        """Defines matrix multiplication"""
        assert isinstance(b, np.ndarray)
        if self.scalar:
            return self.dot_scalar(b)
        return self.dot_matrix(b)

    def dot_matrix(self, b):
        if self.axis == 0:
            return self.value @ b
        else:
            rv = self.value @ np.swapaxes(b, self.axis, 0)
            return np.swapaxes(rv, self.axis, 0)

    def dot_scalar(self, b):
        return self.value * b

    @memoized
    def transpose(self):
        """Defines matrix transpose"""
        if self.scalar:
            return self.value
        return self.value.transpose()

    @property
    def T(self):
        return self.transpose()

    def tosparse(self, type="csr"):
        if not sp.issparse(self.value):
            self.sparse = True
            self.value = tosparse(self.value, type)

    def toarray(self):
        if sp.issparse(self.value):
            self.sparse = False
            self.value = self.value.toarray()
