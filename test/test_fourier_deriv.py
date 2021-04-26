import numpy as np
from pypde.bases import *
import unittest
from test.timer import timeit 

N = 6    # Grid size
RTOL = 1e-5 # np.allclose tolerance

class TestFourier(unittest.TestCase):

    def fun(self,x):
        return np.sin(2*x)

    def usol(self,x):
        return -np.sin(2*x)*(2)**-2

    def setUp(self):
        self.C = Fourier(N)
        self.x = self.C.x

        # -- RHS ---
        self.f = self.fun(self.x)
        self.fhat = self.C.forward_fft(self.f)

        # -- Solution ----
        self.sol = self.usol(self.x)

    @classmethod
    def setUpClass(cls):
        print("-----------------------------")
        print(" Test: Fourier Derivative    ")
        print("-----------------------------")


    def test_solve(self):
        print("\n ** Solve with lstsq **  ")

        D2 = self.C.D(2)

        # Ax = b
        A = D2 
        b = self.fhat[:]

        # Pure Fourier will be singular
        uhat = np.linalg.lstsq(A,b,rcond=None)[0]
        u = self.C.backward_fft(uhat)

        # Remove const part
        u -= u[0]


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
        print("\n ** Solve **  ")

        B = self.C.B(2)

        # x = B@b
        b = self.fhat[:]

        # Pure Fourier will be singular
        uhat = np.zeros(b.shape, dtype=np.complex_)
        uhat[1:] = b[1:]*np.diag(B)[1:]
        u = self.C.backward_fft(uhat)

        # Remove const part
        u -= u[0]

        norm = np.linalg.norm( u-self.sol )
        print(" |u - u_sol|: {:5.2e}"
            .format(norm))

        assert np.allclose(u,self.sol, rtol=RTOL)

