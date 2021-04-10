import numpy as np 
from scipy.sparse import csr_matrix, csc_matrix 
from .inner import inner
from ..utils.memoize import memoized

class Spectralbase():
    ''' 
    Baseclass for Chebyshev Spectral Basesclasses.
    This class defines rudimentary (brute force) inner products
    of Base and Trialfunctions based on get_basis and get_basis_derivative
    methods. 
    Important inner products like mass and stiffness can be overwritten
    in Childclasses with more accurate definitions.  

    Parameters:
        N: int
            Number of grid points
        x: array of floats
            Coordinates of grid points
    '''
    def __init__(self,N,x):
        self.N = N
        self._x = x
        self.name = self.__class__.__name__

    @property
    def x(self):
        return self._x

    def _inner(self,TestFunction=None,k=0):
        ''' 
        Inner Product <Ti^k*Uj> Basefunction T with Testfunction U
        and derivative k
            k = 0: Mass matrix
            k = 1: ? name
            k = 2: Stiffness matrix
        '''
        if TestFunction is None: TestFunction=self
        if k==0:
            return self._to_sparse( 
            inner( self.iter_basis(), 
                    TestFunction.iter_basis(), N = self.N)  )
        else:
            return self._to_sparse( 
            inner( self.iter_deriv(k=k), 
                    TestFunction.iter_basis( ), N = self.N)  )

    def _mass(self,TestFunction=None):
        ''' 
        Mass <TiTj> of Cheby Gauss Lobatto Quad, equvalent to inner(self,self)
        Can be overwritten with more exact inner product
        '''
        return self._inner(TestFunction,k=0)

    def _mass_inv(self):
        return inv(self._mass())

    def _stiff(self,TestFunction=None):
        ''' 
        Stiffness matrix <Ti''Tj> (Inner Product of second derivative with basis) 
        Can be overwritten with more exact inner product
        '''
        return self._inner(TestFunction,k=2)

    def _to_sparse(self,A,tol=1e-12,type="csc"):
        A[np.abs(A)<tol] = 0 # Set close zero elements to zero
        if type in "csc": return csc_matrix(A)
        if type in "csr": return csr_matrix(A)

    def project(self,f):
        ''' Transform to spectral space:
        cn = <Ti,Tj>^-1 @ <Tj,f> where <Ti,Tj> is (sparse) mass matrix'''
        c,sl = np.zeros(self.N), self.slice()
        c[sl] = self._mass_inv()@inner(self.iter_basis(),f)
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