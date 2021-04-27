import numpy as np
from pypde.bases import *
import unittest
from test.timer import timeit 


N = 40    # Grid size
RTOL = 1e-5 # np.allclose tolerance

class TestChebyshev(unittest.TestCase):

    def fun(self,x):
        return np.cos(1*np.pi/2*x)

    def usol(self,x):
        return -np.cos(1*np.pi/2*x)*(1*np.pi/2)**-2

    def setUp(self):
        self.C = Chebyshev(N)
        self.x = self.C.x

        # -- RHS ---
        self.f = self.fun(self.x)
        self.fhat = self.C.forward_fft(self.f)

        # -- Solution ----
        self.sol = self.usol(self.x)

    @classmethod
    def setUpClass(cls):
        print("-----------------------------")
        print(" Test: Chebyshev Derivative  ")
        print("-----------------------------")

    def test_solve(self):
        print("\n ** Solve with lstsq **  ")

        D2 = self.C.D(2)

        # Ax = b
        A = D2 
        b = self.fhat[:]

        # Pure Chebyshev will be singular
        uhat = np.linalg.lstsq(A,b,rcond=None)[0]
        u = self.C.backward_fft(uhat)

        # Remove const part
        u -= u[0]


        norm = np.linalg.norm( u-self.sol )
        print(" |u - u_sol|: {:5.2e}"
            .format(norm))

        assert np.allclose(u,self.sol, rtol=RTOL)


    def test_inverse(self):
        print("\n ** Solve with Pseudoinverse **  ")

        #D2 = self.C.D(2)
        B  = self.C.B(2)
        
        # Pure Chebyshev will be singular
        #A = B@D2 
        A = self.C.I(True)
        b = B@self.fhat[:]

        #uhat = np.linalg.lstsq(A,b,rcond=None)[0]
        # A is diagonal
        uhat = np.zeros(self.fhat.size)
        uhat[2:] = b[:]/np.diag(A[:,2:])
        u = self.C.backward_fft(uhat)

        # Remove const part
        u -= u[0]

        norm = np.linalg.norm( u-self.sol )
        print(" |u - u_sol|: {:5.2e}"
            .format(norm))

        assert np.allclose(u,self.sol, rtol=RTOL)

        # import matplotlib.pyplot as plt
        # from pypde.bases.utils import to_sparse
        # A = to_sparse(A).toarray()
        # plt.spy(A)
        # plt.show()



        # import matplotlib.pyplot as plt
        # x = self.x
        # plt.plot(x,u,"k")
        # plt.plot(x,self.sol,"r--")
        # plt.show()


class TestChebDirichlet(unittest.TestCase):

    def fun(self,x):
        return np.cos(1*np.pi/2*x)

    def usol(self,x):
        return -np.cos(1*np.pi/2*x)*(1*np.pi/2)**-2

    def setUp(self):
        self.C = ChebDirichlet(N)
        self.CH = Chebyshev(N)
        self.x = self.C.x

        # -- RHS ---
        self.f = self.fun(self.x)
        self.fhat = self.CH.forward_fft(self.f)

        # -- Solution ----
        self.sol = self.usol(self.x)

    @classmethod
    def setUpClass(cls):
        print("---------------------------------")
        print(" Test: ChebDirichlet Derivative  ")
        print("---------------------------------")

    def test_solve(self):
        print("\n ** Solve with lstsq **  ")

        S  = self.C.S
        D2 = self.C.D(2)@S

        # Ax = b
        A = D2 
        b = self.fhat[:]

        # Pure Chebyshev will be singular
        uhat = np.linalg.lstsq(A,b,rcond=None)[0]
        u = self.C.backward_fft(uhat)

        norm = np.linalg.norm( u-self.sol )
        print(" |u - u_sol|: {:5.2e}"
            .format(norm))

        # import matplotlib.pyplot as plt
        # x = self.x
        # plt.plot(x,u,"k")
        # plt.plot(x,self.sol,"r--")
        # plt.show()

        assert np.allclose(u,self.sol, rtol=RTOL)


    def test_inv(self):
        print("\n ** Solve with Pseudoinverse **  ")
        from pypde.solver.fortran import linalg as lafort
        _tdma = lafort.tridiagonal.solve_tdma

        B  = self.C.B(2)
        A = self.C.I()@self.C.S
        b = B@self.fhat

        # A is diagonal with elements on 0,+2
        d, u1 = np.diag(A),np.diag(A,+2)
        _tdma(d,u1,b)
        u = self.C.backward_fft(b)

        norm = np.linalg.norm( u-self.sol )
        print(" |u - u_sol|: {:5.2e}"
            .format(norm))
        assert np.allclose(u,self.sol, rtol=RTOL)