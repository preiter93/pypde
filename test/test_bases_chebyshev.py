import numpy as np
from pypde.bases.chebyshev import *
import unittest

N = 100    # Grid size

class TestBasesChebyshev(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("------------------------")
        print(" Test: Chebyhsev Basis  ")
        print("------------------------")

    def test_chebyshev_project(self):
        print("\n  ****** Project ******  ")
        from numpy.polynomial.chebyshev import Chebyshev as Chebnumpy
        # Test projection
        CD = Chebyshev(N)
        x = CD.x
        y = np.sin(2*np.pi/2*x)+0.2#+np.cos(2*np.pi/2*x)
        yhat = CD.project(y)

        # Compate with numpy chebyshev
        cn = Chebnumpy(yhat)
        norm = np.linalg.norm( cn(x)-y )

        print("|pypde - numpy|: {:5.2e}"
            .format(norm))

        # import matplotlib.pyplot as plt 
        # plt.plot(x,y,"k")
        # plt.plot(x,cn(x),"r--")
        # plt.show()

        assert norm<1e-08

    def test_chebyshev_project_mm(self):
        print("\n  ** Project via MassMatrix ** ")
        from numpy.polynomial.chebyshev import Chebyshev as Chebnumpy
        # Test projection
        CD = Chebyshev(N)
        x = CD.x
        y = np.sin(2*np.pi/2*x)+0.2#+np.cos(2*np.pi/2*x)
        yhat = CD.project_via_mass(y)

        # Compate with numpy chebyshev
        cn = Chebnumpy(yhat)
        norm = np.linalg.norm( cn(x)-y )

        print("|pypde - numpy|: {:5.2e}"
            .format(norm))

        # import matplotlib.pyplot as plt 
        # plt.plot(x,y,"k")
        # plt.plot(x,cn(x),"r--")
        # plt.show()

        assert norm<1e-08

    def test_chebyshev_dct(self):
        print("\n ** Forward & Backward via DCT **  ")
        from numpy.polynomial.chebyshev import Chebyshev as Chebnumpy
        # Test projection
        CD = Chebyshev(N)
        x = CD.x
        y = np.sin(2*np.pi/2*x)+np.cos(2*np.pi/2*x)
        yhat = CD.forward_fft(y)
        y = CD.backward_fft(yhat)

        # Compate with numpy chebyshev
        cn = Chebnumpy(yhat)
        norm = np.linalg.norm( cn(x)-y )
        
        print("|pypde - numpy|: {:5.2e}"
            .format(norm))

        assert norm<1e-08

    def test_chebyshev_derivative1(self):
        print("\n ** 1.Derivative **  ")
        deriv = 1
        # Test projection
        CD = Chebyshev(N)
        x = CD.x
        arg = 2*np.pi/2
        y = np.sin(arg*x)+np.cos(arg*x)
        dyc = CD.derivative(y,deriv)
        dym = CD.derivative(y,deriv,method="dm")

        # Compate with analytical solution chebyshev
        dy = arg**deriv*np.cos(arg*x)-arg**deriv*np.sin(arg*x)
        norm = np.linalg.norm( dyc-dy )
        print("fft |pypde - analytical|: {:5.2e}"
            .format(norm))
        norm = np.linalg.norm( dym-dy )
        print("dm  |pypde - analytical|: {:5.2e}"
            .format(norm))
        assert norm<1e-08

    def test_chebyshev_derivative2(self):
        print("\n ** 2.Derivative **  ")
        deriv = 2
        # Test projection
        CD = Chebyshev(N)
        x = CD.x
        arg = 2*np.pi/2
        y = np.sin(arg*x)+np.cos(arg*x)
        dyc = CD.derivative(y,deriv,method="fft")
        dym = CD.derivative(y,deriv,method="dm")

        # Compate with analytical solution chebyshev
        dy = -arg**deriv*np.sin(arg*x)-arg**deriv*np.cos(arg*x)
        norm = np.linalg.norm( dyc-dy )
        print("fft |pypde - analytical|: {:5.2e}"
            .format(norm))
        norm = np.linalg.norm( dym-dy )
        print("dm  |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert norm<1e-08