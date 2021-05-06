import numpy as np
from pypde import *
import unittest

N = 50
LAM = 1/np.pi**2 
RTOL = 1e-3 # np.allclose tolerance

class Diffusion1d(Integrator):
    CONFIG={
        "N": 50,
        "lam": LAM,
        "tsave": 0.01,
        "ndim": 1,
    }
    def __init__(self,**kwargs):
        Integrator.__init__(self)
        self.__dict__.update(**self.CONFIG)
        self.__dict__.update(**kwargs)
        # Field
        shape = (self.N)
        self.field = Field(shape,("CD"))
        # Solver
        self.setup_solver()

    def setup_solver(self):
        # --- Matrices ----
        Sx = self.field.xs[0].S_sp
        Bx = self.field.xs[0].family.B(2,2)
        Ix = self.field.xs[0].family.I(2)
        Ax =  Bx@Sx-self.lam*Ix@Sx
        
        # --- Solver Plans ---
        solver = SolverPlan()
        solver.add_rhs(PlanRHS(Bx,ndim=1,axis=0))    # rhs
        solver.add_lhs(PlanLHS(Ax,ndim=1,axis=0,method="fdma") ) #lhs
        #solver.show_plan()

        self.solver = solver
        self.Sx = Sx

    # @property
    # def _f(self):
    #     ''' Forcing'''
    #     return np.cos(np.pi/2*self.field.x)

    # @property
    # @memoized
    # def _fhat(self):
    #     return self.field.xs[0].family.forward_fft(self._f)

    def update(self,fhat):
        # Solve
        rhs  = self.solver.solve_rhs(self.dt*fhat)
        self.field.vhat[:] = self.solver.solve_lhs(rhs)




class TestHHoltz1D(unittest.TestCase):

    def _f(self,x,lam=LAM):
        return  (1.0+lam*np.pi**2/4)*np.cos(np.pi/2*x)

    def _fsol(self,x):
        return np.cos(np.pi/2*x)

    def setUp(self):
        self.D = Diffusion1d(N=50,dt=1.0,tsave=None)
        self.f = self._f(self.D.field.x)
        self.fhat = self.D.field.xs[0].family.forward_fft(self.f)

    @classmethod
    def setUpClass(cls):
        print("------------------------")
        print(" Test: HHoltz1D         ")
        print("------------------------")

    def test(self):
        self.D.update(self.fhat)
        self.D.field.backward()
        f = self.D.field.v 

        norm = np.linalg.norm( f-self._fsol(self.D.field.x) )

        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(f,self._fsol(self.D.field.x), rtol=RTOL)
        # # -- Plot
        # import matplotlib.pyplot as plt
        # plt.plot(self.D.field.x, self._fsol(self.D.field.x))
        # plt.plot(self.D.field.x, f,"r--")
        # plt.show()