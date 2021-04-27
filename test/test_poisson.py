import numpy as np
from pypde.bases.chebyshev import *
import unittest
from test.timer import timeit 
from numpy.linalg import solve
from pypde.solver.matrix import *
from pypde.solver.operator import *
from pypde.field import SpectralField

N = 50    # Grid size
RTOL = 1e-3 # np.allclose tolerance

# ---------------------------------------------------------
#                           1D
# ---------------------------------------------------------

class TestPoisson(unittest.TestCase):
    def setUp(self):
        self.CD = ChebDirichlet(N)

    def _f(self,x):
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
        cls.u = SpectralField(cls.N+2, "ChebDirichlet")
        cls.x = cls.u.x
        cls.f = SpectralField(cls.N+2, "ChebDirichlet")

        S = cls.u.xs[0].S
        B = cls.u.xs[0].B(2)@S
        A = cls.u.xs[0].I()@S

        # -- RHS ---
        cls.f.v = cls._f(cls,cls.x)
        cls.f.forward()
        fhat = cls.f.vhat
        cls.b = RHSExplicit(f=(B@fhat))

        # -- LHS ---
        
        A0 = MatrixLHS(A,ndim=1,axis=0, solver="solve")                 
        cls.A0 = LHSImplicit(A0)            # Use numpy.linalg.solve

        A1 = MatrixLHS(A,ndim=1,axis=0, solver="tdma")               
        cls.A1 = LHSImplicit(A1)            # Exploit bandedness

        # -- Solution ----
        cls.sol = cls.usol(cls,cls.x)
        cls.solhat = cls.u.forward_fft(cls.sol)

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

# ---------------------------------------------------------
#                           2D
# ---------------------------------------------------------

ARG = np.pi/2
class TestPoisson2D(unittest.TestCase):
    def setUp(self):
        self.CD = ChebDirichlet(N)

    def _f(self,xx,yy):
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
        cls.N  = N
        shape = (N+2,N+2)
        
        # -- u
        cls.u = SpectralField(shape, ("CD","CD"))
        cls.x,cls.y = cls.u.x,cls.u.y
        xx, yy = np.meshgrid(cls.x,cls.y,indexing="ij")
        # -- f
        cls.f = SpectralField(shape, ("CD","CD"))
        cls.f.v = cls._f(cls,xx,yy)
        cls.f.forward()

        # -- Matrices
        # Dx =  cls.u.xs[0].D(2) 
        Sx =  cls.u.xs[0].S
        Bx =  cls.u.xs[0].B(2)@Sx
        Ax =  cls.u.xs[0].I()@Sx


        # -- Eigendecomposition ---
        Sy =  cls.u.xs[1].S
        By =  cls.u.xs[1].B(2)@Sy
        Ay =  cls.u.xs[1].I()@Sy
        #print(Cy.shape)
        ByI = np.linalg.inv(By)
        wy,Qy,Qyi = cls.eigdecomp(cls,Ay.T@ByI.T)

        # -- RHS ---
        fhat = cls.f.vhat
        cls.b = RHSExplicit(f=fhat)
        cls.b.add_PM(MatrixRHS(Bx,axis=0))
        cls.b.add_PM(MatrixRHS(Qy.T,axis=1))

        # -- LHS ---
        Ax = MatrixLHS(A=Bx,ndim=2,axis=0,
            solver="poisson",lam=wy,C=Ax)
        Ay = MatrixLHS(A=Qyi.T,ndim=2,axis=1,
            solver="matmul")
        cls.A = LHSImplicit(Ax)
        cls.A.add(Ay)

        # -- Solution ----
        cls.sol = cls.usol(cls,xx,yy)
        cls.solhat = cls.u.forward_fft(cls.sol)

        print("Initialization finished.")  

    @timeit
    def test_solve(self):
        print("\n ** Numpy Solve **  ")

        # -- Solve ---
        uhat = self.A.solve(self.b.rhs)
        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        # import matplotlib.pyplot as plt
        # u = self.u.backward_fft(uhat)
        # xx,yy = np.meshgrid(self.x,self.x)
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(xx,yy,u, rstride=1, cstride=1, cmap="viridis",edgecolor="k")
        # #ax.plot_surface(xx,yy,self.sol)
        # plt.show()

        assert np.allclose(uhat,self.solhat, rtol=RTOL)

    def eigdecomp(self,A):
        from pypde.utils import eigen_decomp
        return eigen_decomp(A)
