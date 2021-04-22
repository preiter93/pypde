import numpy as np
from pypde.bases.chebyshev import *
from pypde.solver.matrix import *
from pypde.solver.operator import *
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



class TestHholtz(unittest.TestCase):
    def setUp(self):
        pass

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
        super(TestHholtz, cls).setUpClass()
        cls.N  = N
        cls.xf = ChebDirichlet(cls.N+2,bc=(0,0))   # Basis in x
        cls.x  = cls.xf.x                          # X-coordinates

        # -- Matrices ------
        D = Chebyshev(cls.N+2).D(2)     # Spectral derivative matrix
        B = Chebyshev(cls.N+2).B(2)     # Pseudo inverse of D
        S = cls.xf.S                    # Transform stencil Galerkin--> Chebyshev
        M =  (B@S)[2:,:]
        D2 = (B@D@S)[2:,:]

        # -- RHS ---
        fhat = cls.xf.forward_fft(cls.f(cls,cls.x))
        cls.b = RHSExplicit(f=fhat)
        cls.b.add_PM(MatrixRHS(M,axis=0))

        # -- LHS ---
        A = M - LAM*D2
        A0 = MatrixLHS(A,ndim=1,axis=0,
            solver="solve")                 # Use numpy.linalg.solve
        A1 = MatrixLHS(A,ndim=1,axis=0,
            solver="fdma")                  # Use pentadiagonal solver
        cls.A0 = LHSImplicit(A0)
        cls.A1 = LHSImplicit(A1)

        # -- Solution ----
        cls.sol = cls.usol(cls,cls.x)
        cls.solhat = cls.xf.forward_fft(cls.sol)

        print("Initialization finished.")  

    @timeit
    def test_solve(self):
        print("\n ** Numpy Solve **  ")

        # -- Solve ---
        uhat = self.A0.solve(self.b.rhs)
        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)

    @timeit
    def test_fdma(self):
        print("\n ** FDMA Solve **  ")

        # -- Solve ---
        uhat = self.A1.solve(self.b.rhs)
        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)



class TestHholtz2D(unittest.TestCase):
    def setUp(self):
        pass

    def fun(self,x,y,lam):
        return  (1.0+2*lam*np.pi**2/4)*self.usol(self,x,y)

    def usol(self,x,y):
        return np.cos(np.pi/2*x)*np.cos(np.pi/2*y)

    @classmethod
    @timeit
    def setUpClass(cls):
        print("------------------------------")
        print("      Solve Helmholtz (2D)    ")
        print("------------------------------")
        super(TestHholtz2D, cls).setUpClass()
        cls.N  = 1000
        cls.lam = 1/np.pi**6
        cls.xf = ChebDirichlet(cls.N+2,bc=(0,0))   # Basis in x
        cls.x  = cls.xf.x                          # X-coordinates
        xx,yy = np.meshgrid(cls.x,cls.x)

        # -- Matrices ------
        D = Chebyshev(cls.N+2).D(2)     # Spectral derivative matrix
        B = Chebyshev(cls.N+2).B(2)     # Pseudo inverse of D
        S = cls.xf.S                    # Transform stencil Galerkin--> Chebyshev
        M =  (B@S)[2:,:]
        D2 = (B@D@S)[2:,:]
        # ----- Without preconditioning -------
        #M  = cls.xf.mass.toarray() 
        #D2  = cls.xf.stiff.toarray() 
        # -------------------------------------

        # -- RHS ---
        cls.f = cls.fun(cls,xx,yy,cls.lam)
        fhat = cls.xf.forward_fft(cls.f)
        fhat = cls.xf.forward_fft(fhat.T).T
        cls.b = RHSExplicit(f=fhat)
        cls.b.add_PM(MatrixRHS(M,axis=0))
        cls.b.add_PM(MatrixRHS(M,axis=1))

        # -- LHS ---
        A = M - cls.lam*D2
        A0 = MatrixLHS(A,ndim=2,axis=0,
            solver="solve")                 # Use numpy.linalg.solve
        B0 = MatrixLHS(A,ndim=2,axis=1,
            solver="solve")                 # Use numpy.linalg.solve
        A1 = MatrixLHS(A,ndim=2,axis=0,
            solver="fdma")                  # Use pentadiagonal solver
        B1 = MatrixLHS(A,ndim=2,axis=1,
            solver="fdma")                  # Use pentadiagonal solver
        cls.A0 = LHSImplicit(A0)
        cls.A0.add(B0)
        cls.A1 = LHSImplicit(A1)
        cls.A1.add(B1)

        # -- Solution
        cls.sol = cls.usol(cls,xx,yy)
        shat = cls.xf.forward_fft(cls.sol)
        shat = cls.xf.forward_fft(shat.T).T
        cls.solhat = shat

        print("Initialization finished.")  

        cls.bb = np.array(cls.b.rhs.copy(),order="F")

    @timeit
    def test_solve(self):
        print("\n ** Numpy Solve **  ")

        # -- Solve ---
        uhat = self.A0.solve(self.b.rhs)
        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        # import matplotlib.pyplot as plt
        # u = self.xf.backward_fft(uhat)
        # u = self.xf.backward_fft(u.T).T
        # xx,yy = np.meshgrid(self.x,self.x)
        # plt.contourf(xx,yy,u)
        # plt.show()

        assert np.allclose(uhat,self.solhat, rtol=RTOL)


    @timeit
    def test_fdma(self):
        print("\n ** FDMA Solve **  ")

        # -- Solve ---
        #b = self.b.rhs.copy()
        #b = np.array(self.b.rhs.copy(),order="F")
        uhat = self.A1.solve(self.bb)
        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))
        assert np.allclose(uhat,self.solhat, rtol=RTOL)

        # import matplotlib.pyplot as plt
        # u = self.xf.backward_fft(uhat)
        # u = self.xf.backward_fft(u.T).T
        # xx,yy = np.meshgrid(self.x,self.x)
        # plt.contourf(xx,yy,u)
        # plt.show()

    @timeit
    def test_colloc(self):
        print("\n ** Solve with collocation **  ")

        CH = Chebyshev(self.N+2)
        D2 = CH.colloc_deriv_mat(2)
        I = np.eye(self.N+2)

        A = I-self.lam*D2
        xx,yy = np.meshgrid(self.x,self.x)
        b = self.f
        # BCs
        A[0,:] = I[0,:] 
        A[-1,:] = I[-1,:] 
        b[[0,-1],:] = 0
        b[:,[0,-1]] = 0

        for _ in range(LOOP):
            # Solve along axis 0
            u = solve(A,b)
            # Solve along axis 1
            u = solve(A,u.T)

        norm = np.linalg.norm( u-self.sol )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(u,self.sol, rtol=RTOL)
