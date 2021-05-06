import numpy as np
from pypde import *
import unittest

shape = (50,50)
LAM = 1/np.pi**6
RTOL = 1e-3 # np.allclose tolerance

class Diffusion2d(Integrator):
    CONFIG={
        "shape": (50,50),
        "bases": ("CD","CN"),
        "lam": LAM,
        "ndim": 2,
    }
    def __init__(self,**kwargs):
        Integrator.__init__(self)
        self.__dict__.update(**self.CONFIG)
        self.__dict__.update(**kwargs)
        # Field
        self.field = Field(self.shape,self.bases)
        # Solver
        self.setup_solver()

    def setup_solver(self):
        # --- Matrices ----
        Sx = self.field.xs[0].S_sp
        Bx = self.field.xs[0].family.B(2,2)
        Ix = self.field.xs[0].family.I(2)
        Ax =  Bx@Sx-self.lam*Ix@Sx

        Sy = self.field.xs[1].S_sp
        By = self.field.xs[1].family.B(2,2)
        Iy = self.field.xs[1].family.I(2)
        Ay =  By@Sy-self.lam*Iy@Sy
        
        
        # --- Solver Plans ---
        solver = SolverPlan()
        solver.add_rhs(PlanRHS(Bx,ndim=2,axis=0))    # rhs
        solver.add_rhs(PlanRHS(By,ndim=2,axis=1))    # rhs
        solver.add_lhs(PlanLHS(Ax,ndim=2,axis=0,method="fdma") ) #lhs
        solver.add_lhs(PlanLHS(Ay,ndim=2,axis=1,method="fdma") ) #lhs
        #solver.show_plan()

        self.solver = solver

    def update(self,fhat):
        # Solve
        rhs  = self.solver.solve_rhs(self.dt*fhat)
        self.field.vhat[:] = self.solver.solve_lhs(rhs)

class TestHHoltz2D(unittest.TestCase):

    def _f(self,xx,yy,lam=LAM):
        return  (1.0+2*lam*np.pi**2/4)*self._fsol(xx,yy)

    def _fsol(self,xx,yy):
        return np.cos(np.pi/2*xx)*np.sin(np.pi/2*yy)

    def setUp(self):
        self.D = Diffusion2d(shape=shape,dt=1.0,tsave=None)
        self.xx,self.yy = np.meshgrid(self.D.field.x,self.D.field.y,indexing="ij")

        # Forcing
        f = Field(shape,("CH","CH"))
        f.v = self._f(self.xx,self.yy)
        f.forward()
        self.f = f.v
        self.fhat = f.vhat

        # Solution
        self.sol = self._fsol(self.xx,self.yy)

    @classmethod
    def setUpClass(cls):
        print("------------------------")
        print(" Test: HHoltz2D         ")
        print("------------------------")

    def test(self):
        self.D.update(self.fhat)
        self.D.field.backward()
        f = self.D.field.v 

        norm = np.linalg.norm( f-self.sol )

        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(f,self.sol, rtol=RTOL)

        # # -- Plot
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot_surface(self.xx,self.yy,f, rstride=1, cstride=1, cmap="viridis",edgecolor="k")
        # ax.plot_surface(self.xx,self.yy,self.sol)
        # plt.show()

        
