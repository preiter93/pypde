import numpy as np
from pypde.bases.chebyshev import *
import unittest
from test.timer import timeit 
from numpy.linalg import solve
import scipy.sparse as sp
import scipy.sparse.linalg as sla

N = 2000    # Grid size
RTOL = 1e-3 # np.allclose tolerance

class TestPoissonCheb(unittest.TestCase):
    def setUp(self):
        self.CD = ChebDirichlet(N)
        # self.x = self.CD.x
        # self.f = -self.f(self.x)
        # self.sol =  self.fsol(self.x)

    def f(self,x):
        return np.cos(1*np.pi/2*x)

    def fsol(self,x):
        return -np.cos(1*np.pi/2*x)*(1*np.pi/2)**-2

    
    @classmethod
    @timeit
    def setUpClass(cls):
        print("------------------------")
        print("     Solve Poisson      ")
        print("------------------------")
        """ 
        Calculate fftws only once to test solver independently
        """
        super(TestPoissonCheb, cls).setUpClass()
        CD = ChebDirichlet(N)
        cls.CD = CD
        x =  CD.x
        I  = CD.mass.toarray()
        D2 = CD.stiff.toarray()
        
        # Spectral space
        cls.rhs = I@CD.forward_fft(cls.f(cls,x))
        cls.lhs = D2
        cls.lhssp = sp.triu(cls.lhs).tocsr()
        cls.solhat = CD.forward_fft(cls.fsol(cls,x))
        print("Initialization finished.")
        
    @timeit
    def test_1d(self):
        print("\n ** 1-D (Solve) **  ")

        uhat = solve(self.lhs,self.rhs)

        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)


    @timeit
    def test_1d_sparse(self):
        print("\n ** 1-D (Sparse) **  ")

        # Solve
        uhat = sla.spsolve(self.lhssp,self.rhs)

        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)


    @timeit
    def test_1d_triangular(self):
        from scipy.linalg import solve_triangular
        print("\n ** 1-D (Triangular) **  ")

        uhat = solve_triangular(self.lhs,self.rhs)

        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)
