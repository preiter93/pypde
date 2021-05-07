import numpy as np
from pypde import *
import unittest

shape = (50,40)
RTOL = 1e-3 # np.allclose tolerance
ARG = np.pi/2

class Poisson2d(Integrator):
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
        from pypde.solver.utils import eigdecomp
        # --- Matrices ----
        u = self.field
        Sx = u.xs[0].S
        Bx = u.xs[0].family.B(2,2)
        Ix = u.xs[0].family.I(2)
        Ax = Ix@Sx
        Cx = Bx@Sx

        Sy = u.xs[1].S
        By = u.xs[1].family.B(2,2)
        Iy = u.xs[1].family.I(2)
        Ay = Iy@Sy
        Cy = By@Sy

        # -- Eigendecomposition ---
        CyI = np.linalg.inv(Cy)
        wy,Qy,QyI = eigdecomp( CyI@Ay )
        if self.singular:
            wy[0] += 1e-20

        Hy = QyI@CyI@By

        # --- Solver Plans ---
        solver = SolverPlan()
        solver.add_rhs(PlanRHS(Bx,ndim=2,axis=0))
        solver.add_rhs(PlanRHS(Hy,ndim=2,axis=1))

        solver.add_lhs( PlanLHS(Ax,alpha=wy,C=Cx,ndim=2,axis=0,method="poisson") ) 
        solver.add_lhs( PlanLHS(Qy,ndim=2,axis=1,method="multiply") ) 

        self.solver = solver

    def solver_from_template(self):
        from pypde.templates.poisson import solverplan_poisson2d
        self.solver = solverplan_poisson2d(self.field.xs,
            singular=self.singular)

    def update(self,fhat):
        rhs  = self.solver.solve_rhs(fhat)
        self.field.vhat[:] = self.solver.solve_lhs(rhs)
        if self.singular:
            self.field.vhat[0] = 0



# ---------------------------------------------------------
#              Dirichlet + Dirichlet
# ---------------------------------------------------------


class TestPoisson2D(unittest.TestCase):

    def _f(self,xx,yy):
        return np.cos(ARG*xx)*np.cos(ARG*yy)

    def _fsol(self,xx,yy):
        return np.cos(ARG*xx)*np.cos(ARG*yy)*-1/ARG**2/2

    def setUp(self):
        self.D = Poisson2d(shape=shape)
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
        print("-----------------------------------------")
        print(" Test: Poisson2D (Dirichlet + Dirichlet) ")
        print("-----------------------------------------")

    def test(self):
        self.D.update(self.fhat)
        self.D.field.backward()
        f = self.D.field.v 

        norm = np.linalg.norm( f-self.sol )

        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(f,self.sol, rtol=RTOL)

    def test_solver_from_template(self):
        self.D.solver_from_template()
        self.D.update(self.fhat)
        self.D.field.backward()
        f = self.D.field.v 

        norm = np.linalg.norm( f-self.sol )

        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(f,self.sol, rtol=RTOL)

# ---------------------------------------------------------
#              Dirichlet + Neumann
# ---------------------------------------------------------


class TestPoisson2DMixed(unittest.TestCase):

    def _f(self,xx,yy):
        return np.cos(ARG*xx)*np.sin(ARG*yy)

    def _fsol(self,xx,yy):
        return np.cos(ARG*xx)*np.sin(ARG*yy)*-1/ARG**2/2

    def setUp(self):
        self.D = Poisson2d(shape=shape,bases=("CD","CN"))
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
        print("-----------------------------------------")
        print(" Test: Poisson2D (Dirichlet + Dirichlet) ")
        print("-----------------------------------------")

    def test(self):
        self.D.update(self.fhat)
        self.D.field.backward()
        f = self.D.field.v 

        norm = np.linalg.norm( f-self.sol )

        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(f,self.sol, rtol=RTOL)

    def test_solver_from_template(self):
        self.D.solver_from_template()
        self.D.update(self.fhat)
        self.D.field.backward()
        f = self.D.field.v 

        norm = np.linalg.norm( f-self.sol )

        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(f,self.sol, rtol=RTOL)

# ---------------------------------------------------------
#              Neumann + Neumann
# ---------------------------------------------------------


class TestPoisson2DNeumann(unittest.TestCase):

    def _f(self,xx,yy):
        return np.sin(ARG*xx)*np.sin(ARG*yy)

    def _fsol(self,xx,yy):
        return np.sin(ARG*xx)*np.sin(ARG*yy)*-1/ARG**2/2

    def setUp(self):
        self.D = Poisson2d(shape=shape,bases=("CN","CN"),singular=True)
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
        print("-----------------------------------------")
        print(" Test: Poisson2D (Dirichlet + Dirichlet) ")
        print("-----------------------------------------")

    def test(self):
        self.D.update(self.fhat)
        self.D.field.backward()
        f = self.D.field.v 

        norm = np.linalg.norm( f-self.sol )

        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(f,self.sol, rtol=RTOL)

    def test_solver_from_template(self):
        self.D.solver_from_template()
        self.D.update(self.fhat)
        self.D.field.backward()
        f = self.D.field.v 

        norm = np.linalg.norm( f-self.sol )

        print(" |pypde - analytical|: {:5.2e}"
            .format(norm))

        assert np.allclose(f,self.sol, rtol=RTOL)