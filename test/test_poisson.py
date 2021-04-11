import numpy as np
from pypde.bases.chebyshev import *
import unittest
from test.timer import timeit 
from numpy.linalg import solve
import scipy.sparse as sp
import scipy.sparse.linalg as sla

N = 1000    # Grid size
RTOL = 1e-3 # np.allclose tolerance

class TestPoissonCheb(unittest.TestCase):
    def setUp(self):
        self.CD = ChebDirichlet(N)
        self.x = self.CD.x
        self.rhs = -self.f(self.x)
        self.sol =  self.fsol(self.x)

    def f(self,x):
        return np.cos(1*np.pi/2*x)

    def fsol(self,x):
        return self.f(x)*(1*np.pi/2)**-2


    @classmethod
    def setUpClass(cls):
        print("------------------------")
        print("     Solve Poisson      ")
        print("------------------------")

    @timeit
    def test_1d(self):
        print("\n ** 1-D (Solve) **  ")
        CD = self.CD
        I  = CD.mass.toarray()
        D2 = CD.stiff.toarray()
        sl = CD.slice()

        # Solve
        uhat = np.zeros(N)
        rhs = CD.forward_fft(self.rhs)
        rhs = I@rhs[sl]
        lhs = D2[sl,sl]

        uhat[sl] = solve(lhs,rhs)
        u = CD.backward_fft(uhat[sl])

        norm = np.linalg.norm( u-self.sol )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(u,self.sol, rtol=RTOL)


    @timeit
    def test_1d_sparse(self):
        print("\n ** 1-D (Sparse) **  ")
        CD = self.CD
        I  = CD.mass.toarray()
        D2 = CD.stiff.toarray()

        sl = CD.slice()

        # Solve
        uhat = np.zeros(N)
        rhs = CD.forward_fft(self.rhs)
        rhs = I@rhs[sl]
        lhs = D2[sl,sl]
        lhs = sp.triu(lhs).tocsr()

        uhat[sl] = sla.spsolve(lhs,rhs)
        u = CD.backward_fft(uhat[sl])

        norm = np.linalg.norm( u-self.sol )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(u,self.sol, rtol=RTOL)

