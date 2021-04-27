import numpy as np
from pypde.bases.chebyshev import *
import unittest
from test.timer import timeit 
from pypde.solver.matrix import *
from pypde.solver.operator import *
from pypde.field import SpectralField

N = 500    # Grid size
RTOL = 1e-3 # np.allclose tolerance
LAM = 1/np.pi**2 

# ---------------------------------------------------------
#							1D
# ---------------------------------------------------------

class TestHelmholtz(unittest.TestCase):
    def setUp(self):
        pass

    def _f(self,x,lam=LAM):
        return  (1.0+lam*np.pi**2/4)*np.cos(np.pi/2*x)

    def usol(self,x):
        return np.cos(np.pi/2*x)

    
    @classmethod
    @timeit
    def setUpClass(cls):
        print("------------------------------------")
        print("   Solve Helmholtz (Dirichlet 1D)   ")
        print("------------------------------------")
        """ 
        Calculate fftws only once to test solver independently
        """
        super(TestHelmholtz, cls).setUpClass()
        cls.N  = N
        cls.u = SpectralField(cls.N+2, "ChebDirichlet")
        cls.x = cls.u.x
        cls.f = SpectralField(cls.N+2, "ChebDirichlet")

        S = cls.u.xs[0].S
        B = cls.u.xs[0].B(2)@S
        I = cls.u.xs[0].I()@S
        A = B - LAM*I
        

        # -- RHS ---
        cls.f.v = cls._f(cls,cls.x)
        cls.f.forward()
        fhat = cls.f.vhat
        cls.b = RHSExplicit(f=(B@fhat))

        # -- LHS ---
        
        A0 = MatrixLHS(A,ndim=1,axis=0, solver="solve")                 
        cls.A0 = LHSImplicit(A0)            # Use numpy.linalg.solve

        A1 = MatrixLHS(A,ndim=1,axis=0, solver="fdma")               
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
#							2D
# ---------------------------------------------------------

class TestHelmholtz2D(unittest.TestCase):
    def setUp(self):
        self.CD = ChebDirichlet(N)

    def _f(self,xx,yy,lam=LAM):
        return  (1.0+2*lam*np.pi**2/4)*self.usol(self,xx,yy)

    def usol(self,xx,yy):
        return np.cos(np.pi/2*xx)*np.cos(np.pi/2*yy)

    
    @classmethod
    @timeit
    def setUpClass(cls):
        print("------------------------------------")
        print("   Solve Helmholtz (Dirichlet 2D)   ")
        print("------------------------------------")
        """ 
        Calculate fftws only once to test solver independently
        """
        lam = 1/np.pi**6

        super(TestHelmholtz2D, cls).setUpClass()
        cls.N  = N
        shape = (N+2,N+2)
        
        # -- u
        cls.u = SpectralField(shape, ("CD","CD"))
        cls.x,cls.y = cls.u.x,cls.u.y
        xx, yy = np.meshgrid(cls.x,cls.y,indexing="ij")
        # -- f
        cls.f = SpectralField(shape, ("CD","CD"))
        cls.f.v = cls._f(cls,xx,yy,lam)
        cls.f.forward()

        # -- Matrices
        # Dx =  cls.u.xs[0].D(2) 
        Sx =  cls.u.xs[0].S
        Bx =  cls.u.xs[0].B(2)@Sx
        Ix =  cls.u.xs[0].I()@Sx
        Ax =  Bx-lam*Ix


        # -- Eigendecomposition ---
        Sy =  cls.u.xs[1].S
        By =  cls.u.xs[1].B(2)@Sy
        Iy =  cls.u.xs[1].I()@Sy
        Ay =  By-lam*Iy
        
        # ByI = np.linalg.inv(By)
        # wy,Qy,Qyi = cls.eigdecomp(cls,Ay.T@ByI.T)

        # -- RHS ---
        fhat = cls.f.vhat
        cls.b = RHSExplicit(f=fhat)
        cls.b.add_PM(MatrixRHS(Bx,axis=0))
        cls.b.add_PM(MatrixRHS(By,axis=1))

        # -- LHS ---
        AAx = MatrixLHS(A=Ax,ndim=2,axis=0,
            solver="solve")
        AAy = MatrixLHS(A=Ay,ndim=2,axis=1,
            solver="solve")
        cls.A = LHSImplicit(AAx)
        cls.A.add(AAy)

        AAx = MatrixLHS(A=Ax,ndim=2,axis=0,
            solver="fdma")
        AAy = MatrixLHS(A=Ay,ndim=2,axis=1,
            solver="fdma")
        cls.A2 = LHSImplicit(AAx)
        cls.A2.add(AAy)

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
        # ax.plot_surface(xx,yy,self.sol)
        # plt.show()

        assert np.allclose(uhat,self.solhat, rtol=RTOL)

    @timeit
    def test_fdma(self):
        print("\n ** FDMA Solve **  ")

        # -- Solve ---
        uhat = self.A2.solve(self.b.rhs)
        norm = np.linalg.norm( uhat-self.solhat )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(uhat,self.solhat, rtol=RTOL)

    @timeit
    def test_colloc(self):
        print("\n ** Solve with collocation **  ")

        lam = 1/np.pi**6
        CH = Chebyshev(self.N+2)
        D2 = CH.dmp_collocation(2)
        I = np.eye(self.N+2)

        A = I-lam*D2
        xx,yy = np.meshgrid(self.x,self.x)
        b = self.f.v
        # BCs
        A[0,:] = I[0,:]; A[-1,:] = I[-1,:] 
        b[[0,-1],:] = 0; b[:,[0,-1]] = 0

        #for _ in range(LOOP):
        u = np.linalg.solve(A,b) # Solve along axis 0
        u = np.linalg.solve(A,u.T).T # Solve along axis 1

        norm = np.linalg.norm( u-self.sol )
        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(u,self.sol, rtol=RTOL)