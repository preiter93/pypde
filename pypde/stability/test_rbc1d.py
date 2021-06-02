import numpy as np
import unittest
from .rbc1d import solve_rbc1d

N = 51  # Grid size


class TestStability(unittest.TestCase):
    def setUp(self):
        pass

    @classmethod
    def setUpClass(cls):
        print("------------------------")
        print(" Test: Stability RBC (1D)  ")
        print(" N: {:4d}".format(N))
        print("------------------------")

    def test_rbc1d(self):
        # Parameters
        Ny = N
        alpha = 3.14
        Pr = 1.0

        Ra = 1700
        evals, evecs = solve_rbc1d(Ny=Ny, Ra=Ra, Pr=Pr, alpha=alpha, plot=False)
        assert np.imag(evals[-1]) < 0.0

        Ra = 1720
        evals, evecs = solve_rbc1d(Ny=Ny, Ra=Ra, Pr=Pr, alpha=alpha, plot=False)
        assert np.imag(evals[-1]) > 0.0
