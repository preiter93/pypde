import numpy as np
from ..field import *
import unittest

N,M = 40,20     # Grid size
RTOL = 1e-3 # np.allclose tolerance

class TestField(unittest.TestCase):

    def setUp(self):
        shape = (N,M)
        xbase = Base(shape[0],"CD")
        ybase = Base(shape[1],"CN")

        self.S = Field([xbase,ybase])

        # Space
        x,y = self.S.x, self.S.y
        self.xx,self.yy = np.meshgrid(x,y,indexing="ij")

    @classmethod
    def setUpClass(cls):
        print("------------------------")
        print(" Test: Field  ")
        print("------------------------")

    def test_transform(self):
        f =  np.sin(np.pi* self.xx)*np.cos(np.pi*self.yy)
        self.S.v = f
        self.S.forward()
        self.S.backward()
        assert np.allclose(f,self.S.v)

    def test_bc_axis0(self):
        f =  np.sin(np.pi* self.xx)*np.cos(np.pi*self.yy)
        fbc =  f + self.xx
        self.S.v = fbc

        # Boundary conditions
        bc = np.zeros((2,M))
        bc[0,:] = -1
        bc[1,:] =  1

        xbase = Base(N,"CD")
        ybase = Base(M,"CN")
        field_bc    = FieldBC([xbase,ybase],axis=0)
        field_bc.add_bc(bc)

        self.S.add_field_bc(field_bc)

        self.S.v = self.S.make_homogeneous()
        assert np.allclose(self.S.v,f)

    def test_bc_axis1(self):
        xbase = Base(N,"CD")
        ybase = Base(M,"CD")

        S = Field([xbase,ybase])

        f =  np.sin(np.pi* self.xx)*np.sin(np.pi*self.yy)
        fbc =  f + self.yy
        S.v = fbc

        # Boundary conditions
        bc = np.zeros((N,2))
        bc[:,0] = -1
        bc[:,1] =  1
        field_bc    = FieldBC([xbase,ybase],axis=1)
        field_bc.add_bc(bc)

        S.add_field_bc(field_bc)

        S.v = S.make_homogeneous()
        assert np.allclose(S.v,f)

    def test_derivative(self):
        xbase = Base(N,"CH")
        ybase = Base(M,"CH")

        DS = Field([xbase,ybase])

        f =  np.sin(np.pi* self.xx)*np.cos(np.pi*self.yy)
        self.df = -np.pi**2*f

        self.S.v = f
        self.S.forward()
        dfhat = self.S.vhat
        dfhat = self.S.derivative(dfhat,0,axis=1)
        dfhat = self.S.derivative(dfhat,2,axis=0)
        
        # dfhat are chebyshev coefficients now
        DS.vhat = dfhat
        DS.backward()

        # from ..plot.wireframe import plot 
        # plot(self.xx,self.yy,C.v)

        assert np.allclose(DS.v,self.df)


