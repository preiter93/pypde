import numpy as np
import unittest

N = 500  # Grid size
RTOL = 1e-3  # np.allclose tolerance


class TestChebyshev(unittest.TestCase):
    def setUp(self):
        pass

    @classmethod
    def setUpClass(cls):
        print("------------------------")
        print(" Test: Linalg  ")
        print(" N: {:4d}".format(N))
        print("------------------------")

    def test_tdma_py(self):
        from ..linalg.tdma import TDMA

        print("\n**** Tridiagonal Solver (Python) ****  ")

        a = np.random.randn(N - 1)
        b = np.random.randn(N)
        c = np.random.randn(N - 1)
        d = np.random.randn(N)

        x = TDMA(a, b, c, d)

        A = np.diag(a, -1) + np.diag(b, 0) + np.diag(c, 1)
        assert np.allclose(x, np.linalg.solve(A, d))
        print("Success")

    def test_tdma_offset_py(self):
        from ..linalg.tdma import TDMA_offset as TDMA

        print("\n**** Tridiagonal Solver with offset (Python) ****  ")

        k = 2  # offset
        a = np.random.randn(N - k)
        b = np.random.randn(N)
        c = np.random.randn(N - k)
        d = np.random.randn(N)

        x = TDMA(a, b, c, d, k)

        A = np.diag(a, -k) + np.diag(b, 0) + np.diag(c, k)
        assert np.allclose(x, np.linalg.solve(A, d))
        print("Success")
