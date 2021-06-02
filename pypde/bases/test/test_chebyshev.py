import numpy as np
from ..chebyshev import *
import unittest
from .timer import timeit

N = 40     # Grid size
RTOL = 1e-3 # np.allclose tolerance
#
class TestChebyshev(unittest.TestCase):

    def setUp(self):
        self.CH = Chebyshev(N)
        self.x = self.CH.x

        # Function
        arg = 2*np.pi/2
        self.y = np.sin(arg*self.x)+np.cos(arg*self.x)

        self.dy2 = (
        	-arg**2*np.sin(arg*self.x)
        	-arg**2*np.cos(arg*self.x))

    @classmethod
    def setUpClass(cls):
        print("------------------------")
        print(" Test: Chebyshev Basis  ")
        print(" N: {:4d}".format(N))
        print("------------------------")

    @timeit
    def test_project(self):
        print("\n******* Project *******  ")
        from numpy.polynomial.chebyshev import Chebyshev as Chebnumpy

        # Test projection
        yhat = self.CH.project(self.y)

        # Compare with numpy chebyshev
        cn = Chebnumpy(yhat)
        norm = np.linalg.norm( cn(self.x)-self.y )

        print("|pypde - numpy|: {:5.2e}"
            .format(norm))

        assert np.allclose(cn(self.x),self.y, rtol=RTOL)


    @timeit
    def test_dct(self):
        print("\n ** Forward & Backward via DCT **  ")
        from numpy.polynomial.chebyshev import Chebyshev as Chebnumpy

        # Project forward and backward
        yhat = self.CH.forward_fft(self.y)
        y    = self.CH.backward_fft(yhat)

        # Compate with numpy chebyshev
        cn = Chebnumpy(yhat)
        norm = np.linalg.norm( cn(self.x)-y )

        print("|pypde - numpy|: {:5.2e}"
            .format(norm))

        assert np.allclose(cn(self.x),y, rtol=RTOL)

    @timeit
    def test_deriv2_dm(self):
        print("\n ** 2.Derivative via Collocation **  ")

        # Test projection
        dy2 = self.CH.derivative_physical(self.y,2,method="dm")

        # Compare with analytical solution
        norm = np.linalg.norm( dy2-self.dy2 )

        print("fft |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(dy2,self.dy2, rtol=RTOL)

    @timeit
    def test_deriv2_fft(self):
        print("\n ** 2.Derivative via DCT **  ")

        # Test projection
        dy2 = self.CH.derivative_physical(self.y,2,method="fft")

        # Compare with analytical solution
        norm = np.linalg.norm( dy2-self.dy2 )

        print("fft |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(dy2,self.dy2, rtol=RTOL)
#

class TestChebyshev2D(unittest.TestCase):

    def setUp(self):
        self.CH = Chebyshev(N)
        self.x = self.CH.x
        self.xx,self.yy = np.meshgrid(self.x,self.x, indexing="ij")

        # Function
        arg = 2*np.pi/2
        self.f = np.sin(arg*self.xx)*np.cos(arg*self.yy)
        self.df2 = - arg**2*np.sin(arg*self.xx)*np.cos(arg*self.yy)

    @classmethod
    def setUpClass(cls):
        print("\n----------------------------")
        print(" Test: Chebyshev Basis (2D) ")
        print(" N: {:4d}".format(N))
        print("----------------------------")


    @timeit
    def test_dct(self):
        print("\n ** Forward & Backward via DCT **  ")

        fcopy = self.f.copy()
        # Project forward and backward
        fhat = self.CH.forward_fft(fcopy)
        fhat = np.swapaxes(fhat,0,1)
        fhat = self.CH.forward_fft(fhat)
        fhat = np.swapaxes(fhat,0,1)


        f = self.CH.backward_fft(fhat)
        f = np.swapaxes(f,0,1)
        f = self.CH.backward_fft(f)
        f = np.swapaxes(f,0,1)

        # Compate in and output
        norm = np.linalg.norm( f-self.f )

        print("|f_after - f_before|: {:5.2e}"
            .format(norm))

        assert np.allclose(f,self.f, rtol=RTOL)


    @timeit
    def test_deriv2(self):
        print("\n ** 2.Derivative **  ")

        # Test projection
        df2 = self.CH.derivative_physical(self.f,2,method="fft")

        # Compare with analytical solution
        norm = np.linalg.norm( df2-self.df2 )

        print("fft |pypde - analytical|: {:5.2e}"
            .format(norm))

        # import matplotlib.pyplot as plt
        # levels = np.linspace(-10,10,10)
        # plt.contourf(self.xx,self.yy,self.df2,levels=levels)
        # plt.show()
        # plt.contourf(self.xx,self.yy,df2,levels=levels)
        # plt.show()

        assert np.allclose(df2[1:-1],self.df2[1:-1], rtol=1e-2)
