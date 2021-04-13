import numpy as np
from pypde.bases.chebyshev import *
from pypde.bases.inner import *
import unittest

N = 500    # Grid size
RTOL = 1e-3 # np.allclose tolerance

class TestInner(unittest.TestCase):
    def setUp(self):
        self.CH = Chebyshev(N)
        self.CD = ChebDirichlet(N)
        self.derivs = [ 
        (0,0), (0,1), (0,2), (0,3), (0,4),
        (1,0), (2,0), (3,0), (4,0)]

    @classmethod
    def setUpClass(cls):
        print("----------------------------------")
        print(" Test: Inner() of FunctionSpaces  ")
        print("----------------------------------")

    def test_chebyshev(self):
        print("\n  *** Chebyshev (Pure) ***  ")

        for D in self.derivs:
            out_derived = self.CH.inner(self.CH,D=D)
            out_dict = self.CH.inner(self.CH,D=D,lookup=True)
            assert np.allclose(out_derived,out_dict, rtol=RTOL)

    def test_chebdirichlet(self):
        print("\n  *** ChebDirichlet (Pure) ***  ")

        for D in self.derivs:
            out_derived = self.CD.inner(self.CD,D=D)
            out_dict = self.CD.inner(self.CD,D=D,lookup=True)
            assert np.allclose(out_derived,out_dict, rtol=RTOL)

    def test_mixed(self):
        print("\n  *** Family Chebyshev (Mixed) ***  ")

        for D in self.derivs:
            out_derived = self.CD.inner(self.CH,D=D)
            out_dict = self.CD.inner(self.CH,D=D,lookup=True)
            assert np.allclose(out_derived,out_dict, rtol=RTOL)

        for D in self.derivs:
            out_derived = self.CH.inner(self.CD,D=D)
            out_dict = self.CH.inner(self.CD,D=D,lookup=True)
            assert np.allclose(out_derived,out_dict, rtol=RTOL)