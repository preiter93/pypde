import numpy as np
from pypde.bases.chebyshev import *
import unittest
from test.timer import timeit 
from numpy.linalg import solve
from pypde.solver.matrix import *
from pypde.solver.operator import *

N = 500    # Grid size
RTOL = 1e-3 # np.allclose tolerance

class TestPoisson(unittest.TestCase):
    def setUp(self):
        self.CD = ChebDirichlet(N)

    def f(self,x):
        return np.cos(1*np.pi/2*x)

    def usol(self,x):
        return -np.cos(1*np.pi/2*x)*(1*np.pi/2)**-2

    
    @classmethod
    @timeit
    def setUpClass(cls):
        print("----------------------------------")
        print("   Solve Poisson (Dirichlet 1D)   ")
        print("----------------------------------")
        """ 
        Calculate fftws only once to test solver independently
        """
        super(TestPoisson, cls).setUpClass()
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
        A =  D2
        A0 = MatrixLHS(A,ndim=1,axis=0,
            solver="solve")                 # Use numpy.linalg.solve
        A1 = MatrixLHS(A,ndim=1,axis=0,
            solver="tdma")                  # Use diagonal solver
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
    def test_tdma(self):
        print("\n ** TDMA Solve **  ")

        # -- Solve ---
        uhat = self.A1.solve(self.b.rhs)
        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)

ARG = np.pi/2
class TestPoisson2D(unittest.TestCase):
    def setUp(self):
        self.CD = ChebDirichlet(N)

    def f(self,xx,yy):
        return np.cos(ARG*xx)*np.cos(ARG*yy)

    def usol(self,xx,yy):
        return np.cos(ARG*xx)*np.cos(ARG*yy)*-1/ARG**2/2

    
    @classmethod
    @timeit
    def setUpClass(cls):
        print("----------------------------------")
        print("   Solve Poisson (Dirichlet 2D)   ")
        print("----------------------------------")
        """ 
        Calculate fftws only once to test solver independently
        """
        super(TestPoisson2D, cls).setUpClass()
        cls.N  = 100
        cls.xf = ChebDirichlet(cls.N+2,bc=(0,0))   # Basis in x
        cls.x  = cls.xf.x                          # X-coordinates
        xx,yy = np.meshgrid(cls.x,cls.x)

        # -- Matrices ------
        D = Chebyshev(cls.N+2).D(2)     # Spectral derivative matrix
        B = Chebyshev(cls.N+2).B(2)     # Pseudo inverse of D
        S = cls.xf.S                    # Transform stencil Galerkin--> Chebyshev
        M =  (B@S)[2:,:]
        D2 = (B@D@S)[2:,:]
        #D2,M = M,D2

        # -- Eigendecomposition ---
        Mi = np.linalg.inv(M)
        w,Q,Qi = cls.eigdecomp(cls,D2.T@Mi.T)

        # -- RHS ---
        fhat = cls.forward(cls,cls.f(cls,xx,yy))
        cls.b = RHSExplicit(f=fhat)
        cls.b.add_PM(MatrixRHS(M,axis=0))
        cls.b.add_PM(MatrixRHS(Q.T,axis=1))

        # -- LHS ---
        A0 = MatrixLHS(A=M,ndim=2,axis=0,
            solver="poisson",lam=w,C=D2)
        B0 = MatrixLHS(A=Qi.T,ndim=2,axis=1,
            solver="matmul")
        cls.A = LHSImplicit(A0)
        cls.A.add(B0)

        # -- Solution ----
        cls.sol = cls.usol(cls,xx,yy)
        cls.solhat = cls.forward(cls,cls.sol)

        print("Initialization finished.")  

    def eigdecomp(self,A):
        w, Q = np.linalg.eig(A)
        argsort = np.argsort(w)
        w = w[argsort]
        Q = Q[:,argsort]
        Qi = np.linalg.inv(Q)
        return w,Q,Qi

    def forward(self,u):
        uhat = self.xf.forward_fft(u)
        return self.xf.forward_fft(uhat.T).T
    
    def backward(self,uhat):
        u = self.xf.backward_fft(uhat)
        return self.xf.backward_fft(u.T).T

    @timeit
    def test_tdma(self):
        print("\n ** TDMA Solve **  ")

        # -- Solve ---
        uhat = self.A.solve(self.b.rhs)
        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        # import matplotlib.pyplot as plt
        #u = self.backward(uhat)
        
        #print(np.amin(u))
        #print(np.amin(self.sol))
        # xx,yy = np.meshgrid(self.x,self.x)
        # plt.contourf(xx,yy,u)
        # plt.show()

        assert np.allclose(uhat,self.solhat, rtol=RTOL)
        

# ARG = np.pi/2
# class TestPoisson2D(unittest.TestCase):
#     def setUp(self):
#         self.CD = ChebDirichlet(N)

#     def f(self,xx,yy):
#         return np.cos(ARG*xx)*np.cos(ARG*yy)

#     def usol(self,xx,yy):
#         return np.cos(ARG*xx)*np.cos(ARG*yy)*-1/ARG**2/2

    
#     @classmethod
#     @timeit
#     def setUpClass(cls):
#         print("----------------------------------")
#         print("   Solve Poisson (Dirichlet 2D)   ")
#         print("----------------------------------")
#         """ 
#         Calculate fftws only once to test solver independently
#         """
#         super(TestPoisson2D, cls).setUpClass()
#         cls.N  = 50
#         cls.xf = ChebDirichlet(cls.N+2,bc=(0,0))   # Basis in x
#         cls.x  = cls.xf.x                          # X-coordinates
#         xx,yy = np.meshgrid(cls.x,cls.x)

#         # -- Matrices ------
#         D = Chebyshev(cls.N+2).D(2)     # Spectral derivative matrix
#         B = Chebyshev(cls.N+2).B(2)     # Pseudo inverse of D
#         S = cls.xf.S                    # Transform stencil Galerkin--> Chebyshev
#         M =  (B@S)[2:,:]
#         D2 = (B@D@S)[2:,:]

#         # -- Eigendecomposition ---
#         Mi = np.linalg.inv(M)
#         w,Q,Qi = cls.eigdecomp(cls,D2@Mi)

#         # -- RHS ---
#         fhat = cls.forward(cls,cls.f(cls,xx,yy))
#         cls.b = RHSExplicit(f=fhat)
#         cls.b.add_PM(MatrixRHS(M,axis=0))
#         cls.b.add_PM(MatrixRHS(Qi,axis=1))

#         # -- LHS ---
#         A0 = MatrixLHS(A=M,ndim=2,axis=0,
#             solver="poisson",lam=w,C=D2)
#         B0 = MatrixLHS(A=Q,ndim=2,axis=1,
#             solver="matmul")
#         cls.A = LHSImplicit(A0)
#         cls.A.add(B0)

#         # -- Solution ----
#         cls.sol = cls.usol(cls,xx,yy)
#         cls.solhat = cls.forward(cls,cls.sol)

#         print("Initialization finished.")  

#     def eigdecomp(self,A):
#         w, Q = np.linalg.eig(A)
#         argsort = np.argsort(w)
#         w = w[argsort]
#         Q = Q[:,argsort]
#         Qi = np.linalg.inv(Q)
#         return w,Q,Qi

#     def forward(self,u):
#         uhat = self.xf.forward_fft(u)
#         return self.xf.forward_fft(uhat.T).T
    
#     def backward(self,uhat):
#         u = self.xf.backward_fft(uhat)
#         return self.xf.backward_fft(u.T).T

#     @timeit
#     def test_tdma(self):
#         print("\n ** TDMA Solve **  ")

#         # -- Solve ---
#         uhat = self.A.solve(self.b.rhs)
#         norm = np.linalg.norm( uhat-self.solhat )
#         print(" |pypde - analytical|: {:5.2e}"
#             .format(norm))

#         # import matplotlib.pyplot as plt
#         u = self.backward(uhat)
        
#         print(np.amin(u))
#         print(np.amin(self.sol))
#         # xx,yy = np.meshgrid(self.x,self.x)
#         # plt.contourf(xx,yy,u)
#         # plt.show()

#         assert np.allclose(uhat,self.solhat, rtol=RTOL)