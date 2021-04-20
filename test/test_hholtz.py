import numpy as np
import matplotlib.pyplot as plt
from pypde.bases.chebyshev import *
from pypde.bases.solver.cython.tdma_c import solve_twodma_c
from pypde.bases.solver.cython.utda_c import solve_triangular_c
from pypde.solver.fortran import linalg as lafort
import unittest
from test.timer import timeit 
from numpy.linalg import solve
import scipy as sp

N = 2000    # Grid size
RTOL = 1e-3 # np.allclose tolerance
LAM = 1/np.pi**2
LOOP = 10
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
        print("------------------------------")
        print("      Solve Helmholtz (1D)    ")
        print("------------------------------")
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

    @timeit
    def test_1d(self):
        print("\n ** 1-D (Solve) **  ")

        for _ in range(LOOP):
            uhat = solve(self.lhs,self.rhs)

        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)

    @timeit
    def test_1d_lu(self):
        # Use LU Decomposition, which decomposes lhs (A)
        # into a diagonally banded (0&+2) matrix and a
        # upper trianguar matrix
        print("\n ** 1-D (PDA + UTA) **  ")

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
        print("\n ** 1-D (PDA + UTA) FORTRAN**  ")

        _triangular = lafort.triangular.solve_1d
        _twodia = lafort.tridiagonal.solve_twodia_1d

        uhat = np.zeros(N)
        for _ in range(LOOP):
            b = self.b.copy()
            _twodia(self.d,self.u1,b,0)
            uhat = _triangular(self.U,b,0)

        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)

    @timeit
    def test_1d_lu_fortran_scipy(self):
        # Use LU Decomposition, which decomposes lhs (A)
        # into a diagonally banded (0&+2) matrix and a
        # upper trianguar matrix
        print("\n ** 1-D (PDA + UTA) FORTRAN + SCIPY**  ")

        _triangular = sp.linalg.solve_triangular
        _twodia = lafort.tridiagonal.solve_twodia_1d

        uhat = np.zeros(N)
        for _ in range(LOOP):
            b = self.b.copy()
            _twodia(self.d,self.u1,b,0)
            uhat = _triangular(self.U,b)

        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)

    @timeit
    def test_1d_lu_cython(self):
        # Use LU Decomposition, which decomposes lhs (A)
        # into a diagonally banded (0&+2) matrix and a
        # upper trianguar matrix
        print("\n ** 1-D (PDA + UTA) CYTHON**  ")

        uhat = np.zeros(self.b.shape)
        for _ in range(LOOP):
            b = self.b.copy()
            solve_twodma_c(self.d,self.u1,b)
            uhat = solve_triangular_c(self.Uc,b)

        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)
    
#N = 1600
#LAM = 1/np.pi**6

class TestHholtzCheb2D(unittest.TestCase):
    def setUp(self):
        self.CD = ChebDirichlet(N)

    def f(self,x,y,lam):
        return  (1.0+2*lam*np.pi**2/4)*self.usol(self,x,y)

    def usol(self,x,y):
        return np.cos(np.pi/2*x)*np.cos(np.pi/2*y)

    @classmethod
    @timeit
    def setUpClass(cls):
        print("------------------------------")
        print("      Solve Helmholtz (2D)    ")
        print("------------------------------")
        """ 
        Calculate fftws only once to test solver independently
        """
        N = 500
        lam = 1/np.pi**6
        super(TestHholtzCheb2D, cls).setUpClass()
        CD = ChebDirichlet(N)
        cls.CD = CD
        x =  CD.x
        cls.x = x
        xx,yy = np.meshgrid(x,x)
    
        # -- Solution
        cls.sol = cls.usol(cls,xx,yy)
        fhat = CD.forward_fft(cls.sol)
        fhat = CD.forward_fft(fhat.T).T
        cls.solhat = fhat
        # plt.contourf(xx,yy,cls.sol)
        # plt.show()

        
        # Store matrices
        I  = CD.mass.toarray()
        D2 = CD.stiff.toarray()

        cls.rhs = CD.forward_fft(cls.f(cls,xx,yy,lam))
        cls.rhs = I@CD.forward_fft(cls.rhs.T).T@I.T
        cls.lhs = I-lam*D2
        print("Initialization finished.")  

        # LU - Decomposition
        cls.b = cls.rhs.copy()
        cls.b = np.array(cls.b,order="F")
        P,L,cls.U = sp.linalg.lu(cls.lhs)  
        cls.d,cls.u1 = np.diag(L), np.diag(L,-2)
        #cls.d,cls.u1 = cls.d.copy(order='C'),cls.u1.copy(order='C')
        #cls.Uc = cls.U.copy(order='C')

        # Collocation
        CH = Chebyshev(N)
        D2 = CH.colloc_deriv_mat(2)
        cls.rhsc = cls.f(cls,xx,yy,lam)
        I = np.eye(N)
        cls.lhsc = I-lam*D2
        cls.lhsc[0,:] = I[0,:] 
        cls.lhsc[-1,:] = I[-1,:] 
        cls.rhsc[[0,-1],:] = 0
        cls.rhsc[:,[0,-1]] = 0


    @timeit
    def test_colloc(self):
        print("\n ** Solve with collocation **  ")

        for _ in range(LOOP):
            # Solve along axis 0
            u = solve(self.lhsc,self.rhsc)
            # Solve along axis 1
            u = solve(self.lhsc,u.T)


        # xx,yy = np.meshgrid(self.x,self.x)
        # plt.contourf(xx,yy,u)
        # plt.show()

        norm = np.linalg.norm( u-self.sol )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(u,self.sol, rtol=RTOL)

    @timeit
    def test_solve(self):
        print("\n ** Numpy Solve **  ")

        for _ in range(LOOP):
            # Solve along axis 0
            uhat = solve(self.lhs,self.rhs)
            # Solve along axis 1
            uhat = solve(self.lhs,uhat.T)



        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)

    @timeit
    def test_fortran_scipy(self):
        print("\n ** Fortran+Scipy **  ")

        #_triangular = lafort.triangular.solve_2d
        _triangular = sp.linalg.solve_triangular
        _twodia = lafort.tridiagonal.solve_twodia_2d

        for _ in range(LOOP):
            # b = np.array(self.b,order="F")
            b = np.array(self.b.copy(),order="F")
            # Solve along axis 0
            _twodia(self.d,self.u1,b,0)
            uhat = _triangular(self.U,b)
            # Solve along axis 1
            _twodia(self.d,self.u1,uhat,1)
            uhat = _triangular(self.U,uhat.T)


        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)


    @timeit
    def test_fortran(self):
        print("\n ** Fortran **  ")

        _triangular = lafort.triangular.solve_2d
        #_triangular = sp.linalg.solve_triangular
        _twodia = lafort.tridiagonal.solve_twodia_2d

        for _ in range(LOOP):
            # b = np.array(self.b,order="F")
            b = np.array(self.b.copy(),order="F")
            # Solve along axis 0
            _twodia(self.d,self.u1,b,0)
            uhat = _triangular(self.U,b,0)
            # Solve along axis 1
            _twodia(self.d,self.u1,uhat,1)
            uhat = _triangular(self.U,uhat,1)


        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)

        # u = self.CD.backward_fft(uhat)
        # u = self.CD.backward_fft(u.T).T
        # xx,yy = np.meshgrid(self.x,self.x)
        # plt.contourf(xx,yy,u)
        # plt.show()
