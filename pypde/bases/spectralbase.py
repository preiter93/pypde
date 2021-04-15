from .inner import inner
import numpy as np 
from .utils import to_sparse
from ..utils.memoize import memoized
from scipy.sparse.linalg import inv as spinv
from scipy.sparse import issparse

class SpectralBase():
    ''' 
    Baseclass for Spectral Bases. All functionspace classes must inherit 
    from it.
    This class defines important inner products of Base and Trialfunctions
    and how to evaluate and iterate over base functions to derive inner
    products

    Parameters:
        N: int
            Number of grid points
        x: array of floats
            Coordinates of grid points
    '''
    def __init__(self,N,x):
        self._N = N
        self._x = x
        self.name = self.__class__.__name__
        # ID for class, each Space should have its own
        self.id = "SB" 

    @property
    def x(self):
        return self._x

    @property
    def N(self):
        ''' 
        Number of grid points in physical space, equal to 
        size of unrestricted Functionspace'
        '''
        return self._N

    @property
    def M(self):
        ''' Size without BC Functions'''
        return len(range(*self.slice().indices(self.N)))
    

    def inner(self,TestFunction=None,D=(0,0),**kwargs):
        ''' 
        Inner Product <Ti^k*Uj> Basefunction T with Testfunction U
        and derivatives D=ku,kv
            D = (0,0): Mass matrix
            D = (0,1): Grad matrix
            D = (0,2): Stiff matrix
        '''
        if TestFunction is None: TestFunction=self
        return inner(self,TestFunction,w="GL",D=D,**kwargs)

    def _mass(self):
        return self._to_sparse( self.inner(self,D=(0,0) ) )
    def _grad(self):
        return self._to_sparse( self.inner(self,D=(0,1) ) )
    def _stiff(self):
        return self._to_sparse( self.inner(self,D=(0,2) ) )

    @property
    @memoized
    def mass(self):
        ''' 
        Mass <TiTj>, equivalent to inner(self,self)
        Can be overwritten with more exact inner product
        '''
        return self._mass()

    @property
    @memoized
    def _mass_inv(self):
        return spinv(self.mass).toarray()

    @property
    @memoized
    def grad(self):
        ''' Gradient matrix <Ti'Tj> '''
        return self._grad()

    @property
    @memoized
    def stiff(self):
        '''  Stiffness matrix <Ti''Tj> '''
        return self._stiff()

    def _to_sparse(self,A,tol=1e-12,format="csc"):
        if not issparse(A):
            return to_sparse(A,tol,format)
        return A

    def project(self,f):
        ''' Transform to spectral space:
        cn = <Ti,Tj>^-1 @ <Tj,f> where <Ti,Tj> is (sparse) mass matrix'''
        c,sl = np.zeros(self.N), self.slice()
        c[sl] = self._mass_inv@inner(self,f)
        return c

    def evaluate(self,c):
        ''' Evaluate f(x) from spectral coefficients c '''
        y = np.zeros(self.N) 
        for i in range(self.N):
            y += c[i]*self.get_basis(i)
        return y

    def slice(self):
        return slice(0, self.N)

    def iter_basis(self,sl=None):
        ''' Return iterator over all bases '''
        if sl is None: sl=self.slice()
        return (self.get_basis(i) 
            for i in range(self.N)[self.slice()])

    def iter_deriv(self,k=0,sl=None):
        ''' Return iterator over all derivatives of '''
        if sl is None: sl=self.slice()
        return (self.get_basis_derivative(i,k) 
            for i in range(self.N)[self.slice()])