import numpy as np
import scipy.sparse as sp
from .matrix import MatrixBase

"""
Base classes to solve the system of equations
    A x = b
where
    A: N x N matrix
    x: N x M
    b: N x M

First the rhs b must be constructed --> PlanRHS
Then the above system can be solved --> PlanLHS
"""


class Flags:
    def __init__(self, **kwargs):
        self.flags = {}
        self.flags.update(**kwargs)


class MetaPlan:
    def __init__(self, A, ndim, axis, **kwargs):
        self.A = A
        self.ndim = ndim
        self.axis = axis
        self.N = A.shape[1]
        self._flags = Flags()
        self.flags.update({"axis": axis})

    def solve(self):
        raise NotImplementedError

    @property
    def flags(self):
        return self._flags.flags

    def _check_b(self, b):
        assert isinstance(b, np.ndarray)
        assert (
            self.ndim == b.ndim
        ), "Dimensionality mismatch: ndim {:4d} |" " b.ndim {:4d}. Check ndim.".format(
            self.ndim, b.ndim
        )
        assert (
            self.N == b.shape[self.axis]
        ), "Shape mismatch. : N {:4d} |" " b.shape[axis] {:4d}. Check axis.".format(
            self.N, b.shape[self.axis]
        )


class PlanRHS(MetaPlan):
    """
    Handles the right side
        M x = A  b

    This class is used only when premultiplication of
    rhs with A is necessary

    Input:
        A: N x N Array
    """

    def __init__(self, A, ndim, axis):
        MetaPlan.__init__(self, A=A, ndim=ndim, axis=axis)
        self.A = MatrixBase(self.A, axis=self.axis, sparse=True)
        self.ndim = ndim
        self.flags.update({"method": "multiply"})

    def solve(self, b):
        self._check_b(b)
        return self.A.dot(b)


def PlanLHS(A, ndim, axis, method, **kwargs):
    """
    Create a plan on how to solve the generalized system of equations
            A x = b
    where the type of plan depends on the sparsity of the left side

    Input:
        A: N x N Array

        ndim: int
            Dimensionality of rhs b

        axis: int
            Axis along which left side acts on b

        method: str (default='solve')

            'numpy' : General solver np.linalg.solver(A,b)

            'twodma': A is banded with diagonals in offsets  0, 2

            'fdma': A is banded with diagonals in offsets -2, 0, 2, 4

            'matmul': Simple matrix multiplication x = A@b, used in
                      2-dimensional domains

            'poisson': Additional arguments have to be specified!
                       See Plan_PoissonChebyshev

    """
    all_method = {
        "numpy": Plan_numpy,
        "twodma": Plan_twodma,
        "fdma": Plan_fdma,
        "poisson": Plan_Poisson,
        "multiply": PlanRHS,
    }
    if not method in all_method.keys():
        raise ValueError(
            "Method name {:} not found in: {:}".format(method, all_method.keys())
        )
    # Return
    return all_method[method](A=A, ndim=ndim, axis=axis, **kwargs)


class Plan_numpy(MetaPlan):
    def __init__(self, A, ndim, axis):
        MetaPlan.__init__(self, A=A, ndim=ndim, axis=axis)
        self.flags.update({"method": "numpy"})

    def solve(self, b):
        """Solve Ax=b"""
        self._check_b(b)
        if self.axis == 0:
            return np.linalg.solve(self.A, b)
        elif self.axis == 1:
            rv = np.linalg.solve(self.A, np.swapaxes(b, self.axis, 0))
            return np.swapaxes(rv, self.axis, 0)


class Plan_twodma(MetaPlan):
    """A x = b
    A is banded with diagonals in offsets  0, 2
    """

    def __init__(self, A, ndim, axis):
        MetaPlan.__init__(self, A=A, ndim=ndim, axis=axis)
        self.flags.update({"method": "twodma"})

        # Extract diagonals
        self.d = np.diag(A, 0)
        self.u = np.diag(A, 2)

    def solve(self, b):
        """Solve Ax=b"""
        self._check_b(b)
        if self.ndim == 1:
            return self.solve_1d(b)
        elif self.ndim == 2:
            return self.solve_2d(b)
        else:
            raise NotImplementedError("Plan_twodma supports only ndim<3.")

    def solve_1d(self, b, copy=False):
        from .linalg.fortran import twodma

        x = b.copy(order="F") if copy else b
        twodma.solve_twodma_1d(self.d, self.u, x)
        return x

    def solve_2d(self, b, copy=False):
        from .linalg.fortran import twodma

        x = b.copy(order="F") if copy else b
        twodma.solve_twodma_2d(self.d, self.u, x, self.axis)
        return x

    @staticmethod
    def TwoDMA_Solve(d, u1, x):
        """Python version (from shenfun)"""
        n = d.shape[0]
        x[n - 1] = x[n - 1] / d[n - 1]
        x[n - 2] = x[n - 2] / d[n - 2]
        for i in range(n - 3, -1, -1):
            x[i] = (x[i] - u1[i] * x[i + 2]) / d[i]


class Plan_fdma(MetaPlan):
    """A x = b
    A is banded with diagonals in offsets  -2, 0, 2, 4
    """

    def __init__(self, A, ndim, axis):
        MetaPlan.__init__(self, A=A, ndim=ndim, axis=axis)
        self.flags.update({"method": "fdma"})

        # Extract diagonals and do a forward sweep
        l = np.diag(A, -2).copy()
        d = np.diag(A, 0).copy()
        u1 = np.diag(A, 2).copy()
        u2 = np.diag(A, 4).copy()
        self.FDMA_LU(l, d, u1, u2)
        self.l, self.d, self.u1, self.u2 = l, d, u1, u2

    def solve(self, b):
        """Solve Ax=b"""
        self._check_b(b)
        if self.ndim == 1:
            return self.solve_1d(b)
        elif self.ndim == 2:
            return self.solve_2d(b)
        else:
            raise NotImplementedError("Plan_fdma supports only ndim<3.")

    def solve_1d(self, b, copy=False):
        from .linalg.fortran import fdma

        x = b.copy(order="F") if copy else b
        # self.FDMA_Solve(self.l,self.d,self.u1,self.u2,x)
        fdma.solve_fdma_1d(self.l, self.d, self.u1, self.u2, x)
        return x

    def solve_2d(self, b, copy=False):
        from .linalg.fortran import fdma

        x = b.copy(order="F") if copy else b
        fdma.solve_fdma_2d(self.l, self.d, self.u1, self.u2, x, self.axis)
        return x

    @staticmethod
    def FDMA_LU(ld, d, u1, u2):
        """Initialize diagonals"""
        n = d.shape[0]
        for i in range(2, n):
            ld[i - 2] = ld[i - 2] / d[i - 2]
            d[i] = d[i] - ld[i - 2] * u1[i - 2]
            if i < n - 2:
                u1[i] = u1[i] - ld[i - 2] * u2[i - 2]

    @staticmethod
    def FDMA_Solve(l, d, u1, u2, x, axis=0):
        """
        Python version (from shenfun)
        d, u1, u2 are the outputs from FDMA_LU
        """
        n = d.shape[0]
        for i in range(2, n):
            x[i] -= l[i - 2] * x[i - 2]

        x[n - 1] = x[n - 1] / d[n - 1]
        x[n - 2] = x[n - 2] / d[n - 2]
        x[n - 3] = (x[n - 3] - u1[n - 3] * x[n - 1]) / d[n - 3]
        x[n - 4] = (x[n - 4] - u1[n - 4] * x[n - 2]) / d[n - 4]
        for i in range(n - 5, -1, -1):
            x[i] = (x[i] - u1[i] * x[i + 2] - u2[i] * x[i + 4]) / d[i]


class Plan_Poisson(MetaPlan):
    """
    Handles 2D Poisson problems (chebyshev) that has the form

        (A + alpha_i*C) x_i = b_i

    A: N x N
        banded with diagonals in offsets  0, 2
    alpha:  M
        Array with eigenvalues
    C: N x N
        banded with diagonals in offsets -2, 0, 2, 4

    Additional arguments can be supplied:
    singular: bool
        if True the first equation
        when alpha==0 is skipped (pure neumann is singular)

    When solved, rhs b must be of size N x M
    """

    def __init__(self, A, alpha, C, ndim, axis, singular=False):
        assert ndim == 2
        # assert alpha.size == m, \
        # "Size of eigenvalue array does not match!"
        MetaPlan.__init__(self, A=A, ndim=ndim, axis=axis)
        self.flags.update({"method": "poisson"})
        self.alpha = alpha
        self.C = C
        self.singular = singular
        # choose default solver
        self.solve = self.solve_fortran
        # self.solve = self.solve_numpy
        # self.solve = self.solve_pdma

    def solve_numpy(self, b):
        # Swap b
        if self.axis != 0:
            b = np.swapaxes(b, self.axis, 0)
        # Egenvalue size must match with second dim of b
        if self.alpha.size != b.shape[1]:
            raise ValueError(
                "Size of eigenvalue {:3} array does not match to b {:3}!".format(
                    self.alpha.size, b.shape[1]
                )
            )

        x = np.zeros(b.shape)
        for i in range(x.shape[1]):
            A = self.A + self.alpha[i] * self.C
            x[:, i] = np.linalg.solve(A, b[:, i])

        if self.axis != 0:
            return np.swapaxes(x, self.axis, 0)
        return x

    def solve_fortran(self, b):
        from .linalg.fortran import fdma

        fdma.solve_fdma_type2(self.A, self.C, self.alpha, b, self.axis, self.singular)
        return b

    def solve_pdma(self, b):
        from .linalg.pdma import PDMA

        # Swap b
        if self.axis != 0:
            b = np.swapaxes(b, self.axis, 0)
        # Egenvalue size must match with second dim of b
        if self.alpha.size != b.shape[1]:
            raise ValueError(
                "Size of eigenvalue {:3} array does not match to b {:3}!".format(
                    self.alpha.size, b.shape[1]
                )
            )

        x = np.zeros(b.shape)
        for i in range(x.shape[1]):
            A = self.A + self.alpha[i] * self.C
            P = PDMA(A)
            P.init()
            xc = b[:, i].copy()
            P.solve(xc)
            x[:, i] = xc

        if self.axis != 0:
            return np.swapaxes(x, self.axis, 0)
        return x
