import numpy as np
from ..chebyshev import *
import unittest
from .timer import timeit 

N = 500     # Grid size
RTOL = 1e-3 # np.allclose tolerance

class TestChebyshev(unittest.TestCase):

    def setUp(self):
        self.CD = ChebNeumann(N)
        self.x = self.CD.x

        # Function 
        arg = np.pi
        self.y = np.cos(arg*self.x)
        self.dy2 = - arg**2*np.cos(arg*self.x)

    @classmethod
    def setUpClass(cls):
        print("------------------------")
        print(" Test: ChebNeumann Basis  ")
        print(" N: {:4d}".format(N))
        print("------------------------")

    @timeit
    def test_dct(self):
        print("\n ** Forward & Backward via DCT **  ")
        from numpy.polynomial.chebyshev import Chebyshev as Chebnumpy
        
        # Project forward and backward
        yhat = self.CD.forward_fft(self.y)
        y    = self.CD.backward_fft(yhat)

        # Compate with numpy chebyshev
        yhat = self.CD._to_chebyshev(yhat) 
        cn = Chebnumpy(yhat)
        norm = np.linalg.norm( cn(self.x)-y )
        
        print("|pypde - numpy|: {:5.2e}"
            .format(norm))

        assert np.allclose(cn(self.x),y, rtol=RTOL)