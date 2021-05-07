import numpy as np
from pypde import *
import unittest

shape = (50,50)
RTOL = 1e-3 # np.allclose tolerance

# ---------------------------------------------------------
#             Using Kronecker Products
# ---------------------------------------------------------

class Poisson2dKron(Integrator):
    CONFIG={
        "shape": (50,50),
        "bases":("CD","CD"),
        "ndim": 2,
        "singular": False,
    }
    def __init__(self,**kwargs):
        Integrator.__init__(self)
        self.__dict__.update(**self.CONFIG)
        self.__dict__.update(**kwargs)
        # Field
        xbase = Base(self.shape[0],self.bases[0])
        ybase = Base(self.shape[1],self.bases[1])
        self.field = Field([xbase,ybase])
        # Solver
        self.setup_solver()

    def setup_solver(self):
        # --- Matrices ----
        Sx = self.field.xs[0].S_sp
        Bx = self.field.xs[0].family.B(2,2)
        Ix = self.field.xs[0].family.I(2)
        Ax = Ix@Sx

        Sy = self.field.xs[1].S_sp
        By = self.field.xs[1].family.B(2,2)
        Iy = self.field.xs[1].family.I(2)
        Ay = Iy@Sy

        A = np.kron(Ax,By@Sy) + np.kron(Bx@Sx,Ay)
        B = np.kron(Bx,By)

        # --- Solver Plans ---
        solver = SolverPlan()
        solver.add_rhs(PlanRHS(B,ndim=1,axis=0))    # rhs
        solver.add_lhs(PlanLHS(A,ndim=1,axis=0,method="numpy") ) #lhs
        #solver.show_plan()

        self.solver = solver

    def update(self,fhat):
        # Solve
        fhat1d = self.to1d(fhat)
        rhs  = self.solver.solve_rhs(fhat1d)
        vhat = self.solver.solve_lhs(rhs)
        self.field.vhat = self.to2d(vhat,self.field.vhat.shape)

    @staticmethod
    def to1d(v):
        assert v.ndim ==2
        return v.flatten()

    @staticmethod
    def to2d(v,shape):
        assert v.ndim == 1
        return np.reshape(v,shape)


class TestPoisson2DKronecker(unittest.TestCase):

    def  _f(self,xx,yy,arg=np.pi/2):
        return np.cos(arg*xx)*np.cos(arg*yy)

    def _fsol(self,xx,yy,arg=np.pi/2):
        return np.cos(arg*xx)*np.cos(arg*yy)*-1/arg**2/2

    def setUp(self):
        self.D = Poisson2dKron(shape=shape)
        self.xx,self.yy = np.meshgrid(self.D.field.x,self.D.field.y,indexing="ij")

        # Forcing
        xbase = Base(shape[0],"CH")
        ybase = Base(shape[1],"CH")
        f = Field([xbase,ybase])
        f.v = self._f(self.xx,self.yy)
        f.forward()
        self.f = f.v
        self.fhat = f.vhat

        self.sol = self._fsol(self.xx,self.yy)

    @classmethod
    def setUpClass(cls):
        print("-------------------------------------------")
        print(" Test: Poisson2D (With Kronecker Products) ")
        print("-------------------------------------------")

    def test(self):
        self.D.update(self.fhat)
        self.D.field.backward()
        f = self.D.field.v 

        norm = np.linalg.norm( f-self.sol )

        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(f,self.sol, rtol=RTOL)