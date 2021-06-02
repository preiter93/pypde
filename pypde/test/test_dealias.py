import numpy as np
from ..field import *
import unittest

N, M = 40, 20  # Grid size
RTOL = 1e-3  # np.allclose tolerance


class TestDealias(unittest.TestCase):
    def setUp(self):
        shape = (N, M)
        xbase = Base(shape[0], "CD", dealias=3 / 2)
        ybase = Base(shape[1], "CN", dealias=3 / 2)

        self.A = Field([xbase, ybase])
        self.B = Field([xbase, ybase])

        # Space
        x, y = self.A.x, self.A.y
        xx, yy = np.meshgrid(x, y, indexing="ij")

        self.A.v = np.sin(np.pi * xx) * np.cos(np.pi * yy)
        self.B.v = -np.sin(np.pi * xx) * np.cos(np.pi * yy)
        self.A.forward()
        self.B.forward()

    @classmethod
    def setUpClass(cls):
        print("------------------------")
        print(" Test: Dealias  ")
        print("------------------------")

    def test(self):
        A_dealiased = self.A.dealias.backward(self.A.vhat)
        B_dealiased = self.B.dealias.backward(self.B.vhat)
        # Assert size physical space
        assert A_dealiased.shape == tuple(np.array([N, M]) * 3 / 2)

        C_dealiased = A_dealiased * B_dealiased
        Chat_dealiased = self.A.dealias.forward(C_dealiased)
        # Assert size coefficients
        assert Chat_dealiased.shape == self.A.vhat.shape

        # Assert dealiased coefficients are approximately
        # equal to undealiased coefficients
        Chat = self.A.forward(self.A.v * self.B.v)
        assert np.allclose(Chat_dealiased, Chat)
