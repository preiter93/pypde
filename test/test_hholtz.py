import numpy as np
from pypde.bases.chebyshev import *
from pypde.bases.solver.tdma import solve_twodma
from pypde.bases.solver.utda import solve_triangular
from pypde.bases.solver.cython.tdma_c import solve_twodma_c
from pypde.bases.solver.cython.utda_c import solve_triangular_c
import unittest
from test.timer import timeit 
from numpy.linalg import solve
import scipy as sp

N = 2000    # Grid size
RTOL = 1e-3 # np.allclose tolerance
LAM = 1/np.pi**2
LOOP = 100
# -------------------------------------------------
#      u(x) - \lambda \nabla^2 u(x) = f(x)
# with $f(x)=(1+\lambda\pi^2/4)u(x)$ the exact solution is:
#               u(x) = cos(\pi/2 x)
# -------------------------------------------------


def TwoDMA_SolveU(d, u1, x):
    ''' 
    d: N
        diagonal
    u1: N-2
        Diagonal with offset -2
    x: array ndim==1
        rhs
    '''
    assert x.ndim == 1, "Use optimized version for multidimensional solve"
    n = d.shape[0]
    x[0] = x[0]/d[0]
    x[1] = x[1]/d[1]
    for i in range(2,n):
        x[i] = (x[i] - u1[i-2]*x[i-2])/d[i]


class TestHholtzCheb(unittest.TestCase):
    def setUp(self):
        self.CD = ChebDirichlet(N)

    def f(self,x,lam=LAM):
        return  (1.0+lam*np.pi**2/4)*np.cos(np.pi/2*x)

    def usol(self,x):
        return np.cos(np.pi/2*x)

    @classmethod
    @timeit
    def setUpClass(cls):
        print("--------------------------")
        print("      Solve Helmholtz     ")
        print("--------------------------")
        """ 
        Calculate fftws only once to test solver independently
        """
        super(TestHholtzCheb, cls).setUpClass()
        CD = ChebDirichlet(N)
        cls.CD = CD
        x =  CD.x
        I  = CD.mass.toarray()
        D2 = CD.stiff.toarray()

        cls.solhat = CD.forward_fft(cls.usol(cls,x))
        
        # lhs and rhs
        cls.rhs = I@CD.forward_fft(cls.f(cls,x))
        cls.lhs = I-LAM*D2
        print("Initialization finished.")  

        # LU - Decomposition
        cls.b = cls.rhs.copy()
        P,L,U = sp.linalg.lu(cls.lhs)  
        cls.d,cls.u1 = np.diag(L), np.diag(L,-2)
        cls.d,cls.u1 = cls.d.copy(order='C'),cls.u1.copy(order='C')
        cls.U = U 
        cls.Uc = cls.U.copy(order='C')
    # @timeit
    # def test_1d(self):
    #     print("\n ** 1-D (Solve) **  ")

    #     for _ in range(LOOP):
    #         uhat = solve(self.lhs,self.rhs)

    #     norm = np.linalg.norm( uhat-self.solhat )
    #     print(" |pypde - analytical|: {:5.2e}"
    #         .format(norm))

    #     assert np.allclose(uhat,self.solhat, rtol=RTOL)

    @timeit
    def test_1d_lu(self):
        # Use LU Decomposition, which decomposes lhs (A)
        # into a diagonally banded (0&+2) matrix and a
        # upper trianguar matrix
        print("\n ** 1-D (LU + PDA + UTA) **  ")

        for _ in range(LOOP):
            b = self.b.copy()
            TwoDMA_SolveU(self.d,self.u1,b)
            uhat = sp.linalg.solve_triangular(self.U,b)

        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)

    @timeit
    def test_1d_lu_fortran(self):
        # Use LU Decomposition, which decomposes lhs (A)
        # into a diagonally banded (0&+2) matrix and a
        # upper trianguar matrix
        print("\n ** 1-D (LU + PDA + UTA) FORTRAN**  ")

        uhat = np.zeros(N)
        for _ in range(LOOP):
            b = self.b.copy()
            solve_twodma(self.d,self.u1,b)
            uhat = solve_triangular(self.U,b,b.size)

        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)

    @timeit
    def test_1d_lu_cython(self):
        # Use LU Decomposition, which decomposes lhs (A)
        # into a diagonally banded (0&+2) matrix and a
        # upper trianguar matrix
        print("\n ** 1-D (LU + PDA + UTA) CYTHON**  ")

        uhat = np.zeros(self.b.shape)
        for _ in range(LOOP):
            b = self.b.copy()
            solve_twodma_c(self.d,self.u1,b)
            uhat = solve_triangular_c(self.Uc,b)

        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)
    
#     @classmethod
#     @timeit
#     def setUpClass(cls):
#         print("------------------------")
#         print("     Solve Poisson      ")
#         print("------------------------")
#         """ 
#         Calculate fftws only once to test solver independently
#         """
#         super(TestPoissonCheb, cls).setUpClass()
#         CD = ChebDirichlet(N)
#         cls.CD = CD
#         x =  CD.x
#         I  = CD.mass.toarray()
#         D2 = CD.stiff.toarray()
        
#         # Spectral space
#         cls.rhs = I@CD.forward_fft(cls.f(cls,x))
#         cls.lhs = D2
#         cls.lhssp = sp.triu(cls.lhs).tocsr()
#         cls.solhat = CD.forward_fft(cls.fsol(cls,x))
#         print("Initialization finished.")
        
#     @timeit
#     def test_1d(self):
#         print("\n ** 1-D (Solve) **  ")

#         uhat = solve(self.lhs,self.rhs)

#         norm = np.linalg.norm( uhat-self.solhat )
#         print(" |pypde - analytical|: {:5.2e}"
#             .format(norm))

#         assert np.allclose(uhat,self.solhat, rtol=RTOL)


#     @timeit
#     def test_1d_sparse(self):
#         print("\n ** 1-D (Sparse) **  ")

#         # Solve
#         uhat = sla.spsolve(self.lhssp,self.rhs)

#         norm = np.linalg.norm( uhat-self.solhat )
#         print(" |pypde - analytical|: {:5.2e}"
#             .format(norm))

#         assert np.allclose(uhat,self.solhat, rtol=RTOL)


#     @timeit
#     def test_1d_triangular(self):
#         from scipy.linalg import solve_triangular
#         print("\n ** 1-D (Triangular) **  ")

#         uhat = solve_triangular(self.lhs,self.rhs)

#         norm = np.linalg.norm( uhat-self.solhat )
#         print(" |pypde - analytical|: {:5.2e}"
#             .format(norm))

#         assert np.allclose(uhat,self.solhat, rtol=RTOL)
