import numpy as np
from pypde.bases.chebyshev import *
import unittest
from test.timer import timeit 


N = 500    # Grid size
RTOL = 1e-3 # np.allclose tolerance

class TestBasesChebyshev(unittest.TestCase):

    def setUp(self):
        self.CD = Chebyshev(N)
        self.x = self.CD.x

    @classmethod
    def setUpClass(cls):
        print("------------------------")
        print(" Test: Chebyshev Basis  ")
        print("------------------------")
        

    @timeit
    def test_chebyshev_project(self):
        print("\n  ****** Project ******  ")
        from numpy.polynomial.chebyshev import Chebyshev as Chebnumpy
        # Test projection
        arg = 2*np.pi/2
        y = np.sin(arg*self.x)+np.cos(arg*self.x)
        yhat = self.CD.project(y)

        # Compare with numpy chebyshev
        cn = Chebnumpy(yhat)
        norm = np.linalg.norm( cn(self.x)-y )

        print("|pypde - numpy|: {:5.2e}"
            .format(norm))

        assert np.allclose(cn(self.x),y, rtol=RTOL)

    @timeit
    def test_chebyshev_dct(self):
        print("\n ** Forward & Backward via DCT **  ")
        from numpy.polynomial.chebyshev import Chebyshev as Chebnumpy
        # Test
        arg = 2*np.pi/2
        y = np.sin(arg*self.x)+np.cos(arg*self.x)
        yhat = self.CD.forward_fft(y)
        y = self.CD.backward_fft(yhat)

        # Compate with numpy chebyshev
        cn = Chebnumpy(yhat)
        norm = np.linalg.norm( cn(self.x)-y )
        
        print("|pypde - numpy|: {:5.2e}"
            .format(norm))

        assert np.allclose(cn(self.x),y, rtol=RTOL)

    def test_chebyshev_derivative1(self):
        print("\n ** 1.Derivative **  ")
        deriv = 1
        # Test projection
        arg = 2*np.pi/2
        y = np.sin(arg*self.x)+np.cos(arg*self.x)
        dyc = self.CD.derivative(y,deriv)
        dym = self.CD.derivative(y,deriv,method="dm")

        # Compate with analytical solution chebyshev
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
    def test_chebyshev_derivative2_fft(self):
        print("\n ** 2.Derivative via DCT **  ")
        deriv = 2
        # Test projection
        arg = 2*np.pi/2
        y = np.sin(arg*self.x)+np.cos(arg*self.x)
        dyc = self.CD.derivative(y,deriv,method="fft")

        # Compate with analytical solution chebyshev
        dy = -arg**deriv*np.sin(arg*self.x)-arg**deriv*np.cos(arg*self.x)
        norm = np.linalg.norm( dyc-dy )
        print("fft |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(dyc,dy, rtol=RTOL)

    @timeit
    def test_chebyshev_derivative2_dm(self):
        print("\n ** 2.Derivative via DM **  ")
        deriv = 2
        # Test projection
        arg = 2*np.pi/2
        y = np.sin(arg*self.x)+np.cos(arg*self.x)
        dym = self.CD.derivative(y,deriv,method="dm")

        # Compate with analytical solution chebyshev
        dy = -arg**deriv*np.sin(arg*self.x)-arg**deriv*np.cos(arg*self.x)
        norm = np.linalg.norm( dym-dy )
        print("dm  |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(dym,dy, rtol=RTOL)

class TestBasesChebyshev2D(unittest.TestCase):

    def setUp(self):
        self.CD = Chebyshev(N)
        self.x = self.CD.x

    @classmethod
    def setUpClass(cls):
        print("----------------------------")
        print(" Test: Chebyshev Basis (2D) ")
        print("----------------------------")


    @timeit
    def test_chebyshev_dct(self):
        print("\n ** Forward & Backward via DCT **  ")
        # Test
        arg = 2*np.pi/2
        y = np.sin(arg*self.x)+np.cos(arg*self.x)
        yy = np.zeros((self.x.size,2))
        yy[:,0] = y;yy[:,1] = y
        yhat = self.CD.forward_fft(yy)
        uu = self.CD.backward_fft(yhat)
        u = uu[:,0]

        # Compare value before and after dcts
        norm = np.linalg.norm( u-y )
        
        print("|pypde - numpy|: {:5.2e}"
            .format(norm))

        assert np.allclose(y,u, rtol=RTOL)
        assert np.allclose(uu[:,0],uu[:,1], rtol=RTOL)

# import matplotlib.pyplot as plt 
# plt.plot(self.x,y,"k")
# plt.plot(self.x,cn(self.x),"r--")
# plt.show()

# import matplotlib.pyplot as plt 
# plt.plot(self.x,y,"k")
# plt.plot(self.x,u,"r--")
# plt.show()

class TestBasesChebDirichlet(unittest.TestCase):

    def setUp(self):
        self.CD = ChebDirichlet(N)
        self.x = self.CD.x

    @classmethod
    def setUpClass(cls):
        print("----------------------------")
        print(" Test: ChebDirichlet Basis  ")
        print("----------------------------")

    @timeit
    def test_chebyshev_dct(self):
        print("\n ** Forward & Backward via DCT **  ")
        from numpy.polynomial.chebyshev import Chebyshev as Chebnumpy
        # Test projection
        arg = 2*np.pi/2
        y = np.sin(arg*self.x)+np.cos(arg*self.x)
        yhat = self.CD.forward_fft(y)
        u = self.CD.backward_fft(yhat)
        
        # Compare with numpy chebyshev
        # Transform to Chebyshev coefficients
        yhat = self.CD._to_chebyshev(yhat) 
        cn = Chebnumpy(yhat)
        norm = np.linalg.norm( cn(self.x)-u )
        
        print("|pypde - numpy|: {:5.2e}"
            .format(norm))

        assert np.allclose(cn(self.x),u, rtol=RTOL)
        assert np.allclose(y[1:-1],u[1:-1], rtol=RTOL)
        assert not np.allclose(y[0],u[0],rtol=RTOL)

    @timeit
    def test_chebyshev_derivative1(self):
        print("\n ** 1.Derivative **  ")
        deriv = 1
        # Test projection
        arg = 2*np.pi/2
        y = np.sin(arg*self.x)#+np.cos(arg*self.x)
        dyc = self.CD.derivative(y,deriv)

        # Compate with analytical solution chebyshev
        dy = arg**deriv*np.cos(arg*self.x)#-arg**deriv*np.sin(arg*self.x)
        norm = np.linalg.norm( dyc-dy )
        print("fft |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(dyc,dy, rtol=RTOL)

    @timeit
    def test_chebyshev_derivative1_fail(self):
        print("\n ** 1.Derivative (should fail) **  ")
        deriv = 1
        # Test projection
        arg = 2*np.pi/2
        y = np.sin(arg*self.x)+np.cos(arg*self.x)
        dyc = self.CD.derivative(y,deriv)

        # Compate with analytical solution chebyshev
        dy = arg**deriv*np.cos(arg*self.x)-arg**deriv*np.sin(arg*self.x)
        norm = np.linalg.norm( dyc-dy )
        print("fft |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert not np.allclose(dyc,dy, rtol=RTOL)

class TestBasesChebDirichlet2D(unittest.TestCase):

    def setUp(self):
        self.CD = ChebDirichlet(N)
        self.x = self.CD.x

    @classmethod
    def setUpClass(cls):
        print("---------------------------------")
        print(" Test: ChebDirichlet Basis (2D) ")
        print("--------------------------------")


    @timeit
    def test_chebyshev_dct(self):
        print("\n ** Forward & Backward via DCT **  ")
        # Test
        arg = 2*np.pi/2
        y = np.sin(arg*self.x)#+np.cos(arg*self.x)
        yy = np.zeros((self.x.size,2))
        yy[:,0] = y;yy[:,1] = y
        yhat = self.CD.forward_fft(yy)
        uu = self.CD.backward_fft(yhat)
        u = uu[:,0]

        # Compare value before and after dcts
        norm = np.linalg.norm( u-y )
        
        print("|pypde - numpy|: {:5.2e}"
            .format(norm))

        assert np.allclose(y,u, rtol=RTOL)
        assert np.allclose(uu[:,0],uu[:,1], rtol=RTOL)

class TestBasesChebNeumann(unittest.TestCase):

    def setUp(self):
        self.CD = ChebNeumann(N)
        self.x = self.CD.x

    @classmethod
    def setUpClass(cls):
        print("----------------------------")
        print(" Test: ChebNeumann Basis  ")
        print("----------------------------")

    @timeit
    def test_chebyshev_dct(self):
        print("\n ** Forward & Backward via DCT **  ")
        from numpy.polynomial.chebyshev import Chebyshev as Chebnumpy
        # Test projection
        arg = 2*np.pi/2
        y = np.sin(arg*self.x)+np.cos(arg*self.x)
        yhat = self.CD.forward_fft(y)
        u = self.CD.backward_fft(yhat)
        
        # Compare with numpy chebyshev
        # Transform to Chebyshev coefficients
        yhat = self.CD._to_chebyshev(yhat) 
        cn = Chebnumpy(yhat)
        norm = np.linalg.norm( cn(self.x)-u )
        
        print("|pypde - numpy|: {:5.2e}"
            .format(norm))

        assert np.allclose(cn(self.x),u, rtol=RTOL)
        assert np.allclose(y[1:-1],u[1:-1], rtol=RTOL)

    @timeit
    def test_chebyshev_derivative1(self):
        print("\n ** 1.Derivative **  ")
        deriv = 1
        # Test projection
        arg = 2*np.pi/2
        y = +np.cos(arg*self.x)
        dyc = self.CD.derivative(y,deriv)

        # Compate with analytical solution chebyshev
        dy = -arg**deriv*np.sin(arg*self.x)#-arg**deriv*np.sin(arg*self.x)
        norm = np.linalg.norm( dyc-dy )
        print("fft |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(dyc,dy, rtol=RTOL)

    @timeit
    def test_chebyshev_derivative1_fail(self):
        print("\n ** 1.Derivative (should fail) **  ")
        deriv = 1
        # Test projection
        arg = 2*np.pi/2
        y = np.sin(arg*self.x)+np.cos(arg*self.x)
        dyc = self.CD.derivative(y,deriv)

        # Compate with analytical solution chebyshev
        dy = arg**deriv*np.cos(arg*self.x)-arg**deriv*np.sin(arg*self.x)
        norm = np.linalg.norm( dyc-dy )
        print("fft |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert not np.allclose(dyc,dy, rtol=RTOL)