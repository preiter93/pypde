import numpy as np
from pypde.bases.fourier import *
import unittest
from test.timer import timeit 


N = 1000    # Grid size
RTOL = 1e-9 # np.allclose tolerance

class TestBasesFourier(unittest.TestCase):

    def setUp(self):
        self.CD = Fourier(N)
        self.x = self.CD.x

    @classmethod
    def setUpClass(cls):
        print("------------------------")
        print(" Test: Fourier Basis    ")
        print("------------------------")
        

    @timeit
    def test_fourier_fft(self):
        print("\n ** Forward & Backward via FFT **  ")
        # Test projection
        arg = 2*np.pi/2
        y = np.sin(arg*self.x)+np.cos(arg*self.x)
        c = self.CD.forward_fft(y)
        u = self.CD.backward_fft(c)
        assert np.allclose(y,u, rtol=RTOL)

    def test_fourier_derivative1(self):
        print("\n ** 1.Derivative **  ")
        deriv = 1
        # Test projection
        arg = 2
        y = np.sin(arg*self.x)+np.cos(arg*self.x)
        dyc = self.CD.derivative(y,deriv)
        dym = self.CD.derivative(y,deriv,method="dm")

        # Compate with analytical solution
        dy = arg**deriv*np.cos(arg*self.x)-arg**deriv*np.sin(arg*self.x)
        norm = np.linalg.norm( dyc-dy )
        print("fft |pypde - analytical|: {:5.2e}"
            .format(norm))
        norm = np.linalg.norm( dym-dy )
        print("dm  |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(dyc,dy, rtol=RTOL)
        assert np.allclose(dym,dy, rtol=RTOL)

    @timeit
    def test_fourier_derivative2_fft(self):
        print("\n ** 2.Derivative via FFT **  ")
        deriv = 2
        # Test projection
        arg = 2
        y = np.sin(arg*self.x)+np.cos(arg*self.x)
        dyc = self.CD.derivative(y,deriv,method="fft")

        # Compate with analytical solution
        dy = -arg**deriv*np.sin(arg*self.x)-arg**deriv*np.cos(arg*self.x)
        norm = np.linalg.norm( dyc-dy )
        print("fft |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(dyc,dy, rtol=RTOL)

    @timeit
    def test_fourier_derivative2_dm(self):
        print("\n ** 2.Derivative via DM **  ")
        deriv = 2
        # Test projection
        arg = 2
        y = np.sin(arg*self.x)+np.cos(arg*self.x)
        dym = self.CD.derivative(y,deriv,method="dm")

        # Compate with analytical solution
        dy = -arg**deriv*np.sin(arg*self.x)-arg**deriv*np.cos(arg*self.x)
        norm = np.linalg.norm( dym-dy )
        print("dm  |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(dym,dy, rtol=RTOL)

# import matplotlib.pyplot as plt 
# plt.plot(self.x,dy,"k")
# plt.plot(self.x,dym,"r--")
# plt.plot(self.x,dyc,"g-")
# plt.show()