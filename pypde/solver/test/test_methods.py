import numpy as np
from pypde.solver.plans import *
import unittest

N = 20  # Grid size
M = 10
RTOL = 1e-15  # np.allclose tolerance


class Test(unittest.TestCase):
    def setUp(self):
        pass

    @classmethod
    def setUpClass(cls):
        print("----------------------------------")
        print(" Test: Plan Methods         ")
        print("----------------------------------")

    def test_solve(self):
        print("\n  *** Plan: numpy ***  ")

        A = np.random.rand(N, N)
        b = np.random.rand(N, M)

        # Axis == 0
        lhs = PlanLHS(A, ndim=2, axis=0, method="numpy")
        x = lhs.solve(b)

        assert np.allclose(x, np.linalg.solve(A, b), rtol=RTOL)

        # Axis == 1
        lhs = PlanLHS(A, ndim=2, axis=1, method="numpy")
        x = lhs.solve(b.T)

        assert np.allclose(x, np.linalg.solve(A, b).T, rtol=RTOL)
        print("Success")

    def test_twodma(self):
        print("\n  *** Plan: twodma ***  ")

        d = np.random.rand(N)
        u = np.random.rand(N - 2)
        A = np.diag(d, 0) + np.diag(u, 2)

        # ndim == 1
        b = np.random.rand(N)
        lhs = PlanLHS(A, ndim=1, axis=0, method="twodma")
        x = b.copy(order="F")
        x = lhs.solve(x)

        assert np.allclose(x, np.linalg.solve(A, b), rtol=RTOL)

        # ndim == 2
        b = np.random.rand(N, M)
        # Axis == 0
        lhs = PlanLHS(A, ndim=2, axis=0, method="twodma")
        x = b.copy(order="F")
        x = lhs.solve(x)

        assert np.allclose(x, np.linalg.solve(A, b), rtol=RTOL)

        # Axis == 1
        lhs = PlanLHS(A, ndim=2, axis=1, method="twodma")
        x = b.T.copy(order="F")
        x = lhs.solve(x)

        assert np.allclose(x, np.linalg.solve(A, b).T, rtol=RTOL)
        print("Success")

    def test_fdma(self):
        print("\n  *** Plan: fdma ***  ")

        l = np.random.rand(N - 2)
        d = np.random.rand(N)
        u1 = np.random.rand(N - 2)
        u2 = np.random.rand(N - 4)
        A = np.diag(d, 0) + np.diag(u1, 2) + np.diag(u2, 4) + np.diag(l, -2)

        # ndim == 1
        b = np.random.rand(N)
        lhs = PlanLHS(A, ndim=1, axis=0, method="fdma")
        x = b.copy(order="F")
        x = lhs.solve(x)

        assert np.allclose(x, np.linalg.solve(A, b), rtol=RTOL)

        # ndim == 2
        b = np.random.rand(N, M)
        # Axis == 0
        lhs = PlanLHS(A, ndim=2, axis=0, method="fdma")
        x = b.copy(order="F")
        x = lhs.solve(x)

        assert np.allclose(x, np.linalg.solve(A, b), rtol=RTOL)

        # Axis == 1
        lhs = PlanLHS(A, ndim=2, axis=1, method="fdma")
        x = b.T.copy(order="F")
        x = lhs.solve(x)

        assert np.allclose(x, np.linalg.solve(A, b).T, rtol=RTOL)
        print("Success")
