import numpy as np
from pypde import *
import unittest

N = 50
LAM = 1/np.pi**2 
RTOL = 1e-3 # np.allclose tolerance

class Poisson1d(Integrator):
    CONFIG={
        "N": 50,
        "bases":("CD"),
        "ndim": 1,
        "singular": False,
    }
    def __init__(self,**kwargs):
        Integrator.__init__(self)
        self.__dict__.update(**self.CONFIG)
        self.__dict__.update(**kwargs)
        # Field
        shape = (self.N)
        self.field = Field(shape,self.bases)
        # Solver
        self.setup_solver()

    def setup_solver(self):
        # --- Matrices ----
        Sx = self.field.xs[0].S_sp
        Bx = self.field.xs[0].family.B(2,2)
        Ix = self.field.xs[0].family.I(2)
        Ax = Ix@Sx
        if self.singular: 
            # Add very small term to make system non-singular
            # Easier than skipping it in the calculation
            Ax[0,0] += 1e-20 

        # --- Solver Plans ---
        solver = SolverPlan()
        solver.add_rhs(PlanRHS(Bx,ndim=1,axis=0))    # rhs
        solver.add_lhs(PlanLHS(Ax,ndim=1,axis=0,method="twodma") ) #lhs
        #solver.show_plan()

        self.solver = solver
        self.Sx = Sx

    def update(self,fhat):
        # Solve
        rhs  = self.solver.solve_rhs(fhat)
        self.field.vhat[:] = self.solver.solve_lhs(rhs)
        if self.singular: 
            self.field.vhat[0] = 0

# ---------------------------------------------------------
#                    Dirichlet
# ---------------------------------------------------------

class TestPoisson1D(unittest.TestCase):

    def _f(self,x):
        return  np.cos(1*np.pi/2*x)

    def _fsol(self,x):
        return -np.cos(1*np.pi/2*x)*(1*np.pi/2)**-2

    def setUp(self):
        self.D = Poisson1d(N=N)
        self.f = self._f(self.D.field.x)
        self.fhat = self.D.field.xs[0].family.forward_fft(self.f)
        self.sol = self._fsol(self.D.field.x)

    @classmethod
    def setUpClass(cls):
        print("-----------------------------")
        print(" Test: Poisson1D (Dirichlet) ")
        print("-----------------------------")

    def test(self):
        self.D.update(self.fhat)
        self.D.field.backward()
        f = self.D.field.v 

        norm = np.linalg.norm( f-self.sol )

        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(f,self.sol, rtol=RTOL)

# ---------------------------------------------------------
#                    Neumann
# ---------------------------------------------------------

class TestPoisson1DNeumann(unittest.TestCase):

    def _f(self,x):
        return  np.sin(1*np.pi/2*x)

    def _fsol(self,x):
        return -np.sin(1*np.pi/2*x)*(1*np.pi/2)**-2

    def setUp(self):
        self.D = Poisson1d(N=N,bases=("CN"),singular=True)
        self.f = self._f(self.D.field.x)
        self.fhat = self.D.field.xs[0].family.forward_fft(self.f)
        self.sol = self._fsol(self.D.field.x)

    @classmethod
    def setUpClass(cls):
        print("---------------------------")
        print(" Test: Poisson1D (Neumann) ")
        print("---------------------------")

    def test(self):
        self.D.update(self.fhat)
        self.D.field.vhat[0] = 0
        self.D.field.backward()
        f = self.D.field.v 

        norm = np.linalg.norm( f-self.sol )

        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(f,self.sol, rtol=RTOL)
        # # -- Plot
        # import matplotlib.pyplot as plt
        # plt.plot(self.D.field.x, self._fsol(self.D.field.x))
        # plt.plot(self.D.field.x, f,"r--")
        # plt.show()