import numpy as np
from .inner import inner
from .utils import tosparse
from .memoize import memoized

# from scipy.sparse.linalg import inv as spinv
from scipy.sparse import issparse

SPARSE = True


def Base(N, key, *args, **kwargs):
    """
    Convenient way to initalize Base classes from key

    Parameters:
        N: int
            Number of grid points
        key: str
            Key (ID) of respective class
    """
    return _bases_from_key(key)(N, *args, **kwargs)


def _bases_from_key(key):
    from .chebyshev import Chebyshev, ChebDirichlet, ChebNeumann, ChebDirichletNeumann
    from .chebyshev import DirichletC, NeumannC
    from .fourier import Fourier

    if key == "CH" or key == "Chebyshev":
        return Chebyshev
    elif key == "CD" or key == "ChebDirichlet":
        return ChebDirichlet
    elif key == "CN" or key == "ChebNeumann":
        return ChebNeumann
    elif key == "CDN" or key == "ChebDirichletNeumann":
        return ChebDirichletNeumann
    elif key == "DC" or key == "DirichletC":
        return DirichletC
    elif key == "NC" or key == "NeumannC":
        return NeumannC
    elif key == "FO" or key == "Fourier":
        return Fourier
    else:
        raise ValueError("Key {:} not available.".format(key))


class MetaBase:
    """
    Metaclass for Spectral Bases. All functionspace classes must inherit
    from it.
    This class defines important inner products of Base and Trialfunctions
    and how to evaluate and iterate over base functions to derive inner
    products

    Parameters:
        N: int
            Number of grid points
        x: array of floats
            Coordinates of grid points
        dealias: float (optional)
            If dealias is not None, a new additional
            space is initialized under self.dealias, which is of
            size N*dealias.
            Use dealias=3/2 to dealias the products in the navier
            stokes equations.
    """

    def __init__(self, N, x, dealias=None):
        self._N = N
        self._x = x
        self.name = self.__class__.__name__

        # ID for class, each Space should have its own
        self.id = None

        if dealias is not None:
            self.create_dealiased_base(self.N * dealias)

    @property
    def x(self):
        return self._x

    @property
    def N(self):
        """Number of grid points in physical space"""
        return self._N

    @property
    def M(self):
        """Number of coefficients without BC"""
        return len(range(*self.slice().indices(self.N)))

    def create_dealiased_base(self, size):
        """Initialize new space for dealiased transforms"""
        self.dealias = self.__class__(int(size), dealias=None)

    # ---------------------------------------------
    #               Inner Products
    # ---------------------------------------------

    def inner(self, TestFunction=None, D=(0, 0), **kwargs):
        """
        Inner Product <Ti^k*Uj> Basefunction T with Testfunction U
        and derivatives D=ku,kv
            D = (0,0): Mass matrix
            D = (0,1): Grad matrix
            D = (0,2): Stiff matrix
        """
        if TestFunction is None:
            TestFunction = self
        return inner(self, TestFunction, w="GL", D=D, **kwargs)

    def _mass(self):
        return self.inner(self, D=(0, 0))

    def _grad(self):
        return self.inner(self, D=(0, 1))

    def _stiff(self):
        return self.inner(self, D=(0, 2))

    @property
    @memoized
    def mass(self):
        """
        Mass <TiTj>, equivalent to inner(self,self)
        Can be overwritten with more exact inner product
        """
        return self._mass()

    @property
    @memoized
    def grad(self):
        """Gradient matrix <Ti'Tj>"""
        return self._grad()

    @property
    @memoized
    def stiff(self):
        """Stiffness matrix <Ti''Tj>"""
        return self._stiff()

    @property
    @memoized
    def mass_sp(self):
        return self._tosparse(self.mass)

    @property
    @memoized
    def grad_sp(self):
        return self._tosparse(self.grad)

    @property
    @memoized
    def stiff_sp(self):
        return self._tosparse(self.stiff)

    def _tosparse(self, A, tol=1e-12, format="csc"):
        if not issparse(A):
            return tosparse(A, tol, format)
        return A

    def solve_mass(self, f):
        """
        Solve Mx = f, where m is mass matrix.
        Can be implemented more efficient in subclasses
        """
        return np.linalg.solve(self.mass, f)

    # ---------------------------------------------
    #     Project/Iterate over basis functions
    # ---------------------------------------------

    def project(self, f):
        """Transform to spectral space:
        cn = <Ti,Tj>^-1 @ <Tj,f> where <Ti,Tj> is (sparse) mass matrix"""
        c, sl = np.zeros(self.N), self.slice()
        c[sl] = self.solve_mass(inner(self, f))
        return c

    def evaluate(self, c):
        """Evaluate f(x) from spectral coefficients c"""
        y = np.zeros(self.N)
        for i in range(self.N):
            y += c[i] * self.get_basis(i)
        return y

    def slice(self):
        return slice(0, self.N)

    def iter_basis(self, sl=None):
        """Return iterator over all bases"""
        if sl is None:
            sl = self.slice()
        return (self.get_basis(i) for i in range(self.N)[self.slice()])

    def iter_deriv(self, k=0, sl=None):
        """Return iterator over all derivatives of"""
        if sl is None:
            sl = self.slice()
        return (self.get_basis_derivative(i, k) for i in range(self.N)[self.slice()])

    # ---------------------------------------------
    #    Short Names for widely used functions
    # ---------------------------------------------

    @property
    @memoized
    def S(self):
        """
        Transformation Galerkin <-> Chebyshev
        Returns Identity matrix if there is no stencil defined
        """
        if hasattr(self, "stencil"):
            return self.stencil()

        return np.eye(self.N)

    @property
    @memoized
    def ST(self):
        """Stencil transpose"""
        if hasattr(self, "stencil"):
            return self.stencil(transpose=True)
        return np.eye(self.N)

    @property
    @memoized
    def S_sp(self):
        """Sparse version of stencil"""
        return self._tosparse(self.S)

    @property
    @memoized
    def ST_sp(self):
        """Sparse version of stencil transpose"""
        return self._tosparse(self.ST)

    def D(self, deriv):
        if deriv == 0:
            return self.mass
        if deriv == 1:
            return self.grad
        if deriv == 2:
            return self.stiff
        raise ValueError("deriv>2 not supported")
