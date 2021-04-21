import numpy as np
from pypde.bases.chebyshev import *
import unittest
from test.timer import timeit 
from numpy.linalg import solve
from pypde.solver.fortran import linalg as lafort

N = 2000    # Grid size
RTOL = 1e-3 # np.allclose tolerance

class TestPoissonChebNPI(unittest.TestCase):
    def setUp(self):
        self.CD = ChebDirichlet(N)

    def f(self,x):
        return np.sin(1*np.pi/2*x)

    def fsol(self,x):
        return -np.sin(1*np.pi/2*x)*(1*np.pi/2)**-2

    
    @classmethod
    @timeit
    def setUpClass(cls):
        print("----------------------------------------------------")
        print("     Solve Poisson (Neumann) - Pseudoinverse      ")
        print("----------------------------------------------------")
        """ 
        Calculate fftws only once to test solver independently
        """
        super(TestPoissonChebNPI, cls).setUpClass()
        CD = ChebNeumann(N)
        CH = Chebyshev(N)
        cls.CD,cls.CD = CD,CH
        x =  CD.x
        D  = CH.spec_deriv_mat(2)
        B  = CH.spec_deriv_mat_inverse(2)
        S  = CD.stencil(True) # Transform stencil

        # Spectral space
        cls.rhs = (B@CH.forward_fft(cls.f(cls,x)))[2:]
        cls.lhs = (B@D@S)[2:,:]

        cls.d = np.diag(cls.lhs,0)
        cls.u1 = np.diag(cls.lhs,2)

        cls.solhat = CD.forward_fft(cls.fsol(cls,x))
        print("Initialization finished.")
        
    @timeit
    def test_1d(self):
        print("\n ** Solve **  ")

        uhat = np.zeros(self.rhs.size)
        uhat[1:] = solve(self.lhs[1:,1:],self.rhs[1:])

        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)

    @timeit
    def test_banded(self):
        print("\n ** Banded **  ")

        _twodia = lafort.tridiagonal.solve_tdma

        uhat = np.zeros(self.rhs.size)
        uhat[1:] = self.rhs[1:]
        _twodia(self.d[1:],self.u1[1:],uhat[1:],0)

        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)