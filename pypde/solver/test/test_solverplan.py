import numpy as np
from pypde.solver.solverplan import *
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
        print(" Test: SolverPlan                 ")
        print("----------------------------------")

    def test_solve(self):
        print("\n  *** A ***  ")

        print("Along axis 0:")
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)
        b = np.random.rand(N, M)

        # Axis == 0
        rhs = PlanRHS(B, ndim=2, axis=0)
        lhs = PlanLHS(A, ndim=2, axis=0, method="numpy")
        solver = SolverPlan()
        solver.add_lhs(lhs)
        solver.add_rhs(rhs)

        solver.show_plan()

        rhs = solver.solve_rhs(b)
        x = solver.solve_lhs(rhs)

        assert np.allclose(x, np.linalg.solve(A, B @ b), rtol=RTOL)

        print("Along axis 1:")
        # Axis == 1
        b = np.random.rand(M, N)
        rhs = PlanRHS(B, ndim=2, axis=1)
        lhs = PlanLHS(A, ndim=2, axis=1, method="numpy")
        solver = SolverPlan()
        solver.add_lhs(lhs)
        solver.add_rhs(rhs)

        solver.show_plan()

        rhs = solver.solve_rhs(b)
        x = solver.solve_lhs(rhs)

        rhs = b @ B.T
        assert np.allclose(x, np.linalg.solve(A, rhs.T).T, rtol=RTOL)
        print("Success")
